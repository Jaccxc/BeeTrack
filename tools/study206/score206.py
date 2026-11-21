"""Return-query bounds, stable group greedy and physical native scoring.

Source/synthetic only until published prereg + owners CPU release + root GO.
No field files are opened on import. Owner certificates supply ONE fixed
earliest legal row per query/threshold; greedy never searches alternative dates.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,hashlib,json,shutil,tempfile
from fractions import Fraction
from numbers import Integral
from pathlib import Path
import numpy as np

THRESHOLDS=('0.5','0.3');ARMS=('K','R05','R03')
def integer(x):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral):raise ValueError('integer required')
    return int(x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x
def object_sha(x):return hashlib.sha256(json.dumps(plain(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_sha(a):
    a=np.ascontiguousarray(a,dtype='<f8')
    return hashlib.sha256(json.dumps(list(a.shape)).encode()+b'\n'+a.tobytes()).hexdigest()
def bins(events,nf):
    """Exact official offsets0..29, first/last partial windows retained."""
    nf=integer(nf)
    if nf<1:raise ValueError('positive full clip clock required')
    result=[]
    for o in range(30):
        part=np.zeros(((nf+25*o+749)//750,2),np.int64)
        for f,d in events:
            f,d=integer(f),integer(d)
            if not 1<=f<=nf or d not in (-1,1):raise ValueError('legal signed event required')
            part[(f-1+25*o)//750,int(d<0)]+=1
        result.append(part)
    return np.vstack(result)
def query_key(q):return q['query_id']
def queries(unit):
    return [(m,q) for m in unit['members'] for q in m['queries']]
def event_column(q,nf):
    e=q['event'];return bins([(e['frame'],e['direction'])],nf)
def candidate(q,t):
    label=q['labels'][t]
    if not label['confirmed']:return None
    c=label['candidate']
    if c is None or c['query_id']!=query_key(q) or not c['event_delete_only']:raise ValueError('confirmed fixed deletion certificate required')
    row=np.asarray(c['added_row'],float)
    if row.shape!=(7,) or not np.isfinite(row).all() or integer(c['frame'])!=row[0]:raise ValueError('candidate clock/row differs')
    return c
def raw_state(unit):
    pred=np.asarray(unit['pred_sum']);gt=np.asarray(unit['gt_bins'])
    if pred.shape!=bins([],unit['nf']).shape or gt.shape!=pred.shape:raise ValueError('full official30phase shape differs')
    if any(not np.issubdtype(a.dtype,np.integer) or (a<0).any() for a in (pred,gt)):raise ValueError('nonnegative integer counts required')
    if int(gt.sum())%30:raise ValueError('GT phase census differs')
    return pred-3*gt

def bound_unit(unit,t=None,eligible_only=False):
    """Optimistic count gain U; windows choose independently, NOT attainable."""
    r=raw_state(unit);capacity=np.zeros_like(r);keys=[];bees=set()
    for m,q in queries(unit):
        if eligible_only and not q['eligible']:continue
        c=None if t is None else candidate(q,t)
        if t is not None and c is None:continue
        col=event_column(q,unit['nf'])
        if c is not None and not np.array_equal(np.asarray(c['delta_windows']),-col):raise ValueError('certificate is not exact original event deletion')
        capacity+=col;keys.append(query_key(q))
        if t is not None:bees.add(object_sha(q['labels'][t]['GT_run_key']))
    if len(keys)!=len(set(keys)):raise ValueError('duplicate query key')
    upper=int(np.minimum(np.maximum(r,0),capacity).sum());baseline=int(np.abs(r).sum())
    return dict(threshold=t,population='confirmed_same_bee_candidates' if t else 'eligible_deployment_queries' if eligible_only else 'all_native_event_roster',
        query_count=len(keys),confirmed_GT_runs=len(bees) if t else None,
        baseline_numerator=baseline,gain_upper_numerator=upper,lower_numerator=baseline-upper,
        denominator=3*int(np.asarray(unit['gt_bins']).sum()),capacity=capacity.tolist(),query_ids=keys,
        caution='Per-window optimistic relaxation; query count is not distinct bees.')

def greedy_group(units,t,joint_validate,on_accept=None):
    """Highest strictly positive whole-group RAW L1 gain, then query_key.

    joint_validate(unit, selected_query_ids, threshold) returns {safe,reason,...}.
    Rejected additions are recorded and never retried; no alternative row/date.
    """
    if not units or len({(u['domain'],u['group']) for u in units})!=1:raise ValueError('one historical three-member group required')
    table={u['unit']:u for u in units};residual={k:raw_state(u) for k,u in table.items()};selected={k:[] for k in table}
    remaining={};steps=[];rejected=[];used_rows=set();baseline=sum(int(np.abs(r).sum()) for r in residual.values())
    for u in units:
        for m,q in queries(u):
            c=candidate(q,t)
            if c is None:continue
            key=query_key(q)
            if key in remaining:raise ValueError('duplicate group query')
            delta=np.asarray(c['delta_windows'])
            if not np.array_equal(delta,-event_column(q,u['nf'])):raise ValueError('fixed candidate changes more than one original event')
            remaining[key]=(u,m,q,c,delta)
    while remaining:
        ranked=[]
        for key,(u,m,q,c,delta) in remaining.items():
            r=residual[u['unit']];gain=int(np.abs(r).sum()-np.abs(r+delta).sum())
            if gain>0:ranked.append((-gain,key))
        if not ranked:break
        neg,key=min(ranked);u,m,q,c,delta=remaining.pop(key);ukey=u['unit'];gain=-neg
        row_resource=(str(m['member']),object_sha(c['GT_row_key']))
        if row_resource in used_rows:audit=dict(safe=False,reason='same_member_GT_row_reused')
        else:audit=joint_validate(u,sorted(selected[ukey]+[key]),t)
        if not isinstance(audit,dict) or 'safe' not in audit:raise ValueError('explicit joint safety proof required')
        if not audit['safe']:
            rejected.append(dict(query_id=key,unit=ukey,proposed_gain=gain,reason=audit['reason'],joint_audit=audit));continue
        before=int(np.abs(residual[ukey]).sum());residual[ukey]+=delta;after=int(np.abs(residual[ukey]).sum())
        selected[ukey].append(key);used_rows.add(row_resource)
        steps.append(dict(step=len(steps),query_id=key,unit=ukey,gain=gain,before=before,after=after,
            fixed_return_frame=c['frame'],gt_row_key=c['GT_row_key'],joint_audit=audit))
        if on_accept is not None:
            steps[-1]['physical_prefix']=on_accept({k:sorted(v) for k,v in selected.items()},t,steps[-1],residual)
    final=sum(int(np.abs(r).sum()) for r in residual.values())
    if baseline-final!=sum(s['gain'] for s in steps):raise AssertionError('greedy telescope differs')
    bees=set()
    for u in units:
        for m,q in queries(u):
            if query_key(q) in selected[u['unit']]:bees.add(object_sha(q['labels'][t]['GT_run_key']))
    return dict(threshold=t,baseline_numerator=baseline,final_numerator=final,gain_numerator=baseline-final,
        selected={k:sorted(v) for k,v in selected.items()},selected_query_count=len(steps),selected_GT_runs=len(bees),
        steps=steps,rejected=rejected,remaining_nonpositive=sorted(remaining),
        witness_pred_sum={k:(r+3*np.asarray(table[k]['gt_bins'])).tolist() for k,r in residual.items()},
        policy='Fixed earliest row;strict positive raw group gain;query_id tie;unsafe proposals permanently rejected.')

def aggregate_bounds(units,reports):
    """Pool actual clip numerators/GT per group, then equal groups per hive."""
    out={}
    for h in sorted({u['domain'] for u in units}):
        groups=[]
        for g in sorted({u['group'] for u in units if u['domain']==h}):
            rows=[reports[u['unit']] for u in units if u['domain']==h and u['group']==g]
            den=sum(r['denominator'] for r in rows)
            if den<=0:raise ValueError('undefined zero-GT WHOLE group')
            gain=sum(r['gain_upper_numerator'] for r in rows)
            groups.append(dict(group=g,denominator=den,gain_upper_numerator=gain,gain_upper_percent=float(Fraction(100*gain,den))))
        exact=sum((Fraction(100*g['gain_upper_numerator'],g['denominator']) for g in groups),Fraction())/len(groups)
        out[h]=dict(groups=groups,gain_upper_percent=float(exact),fraction=[exact.numerator,exact.denominator])
    return out

def necessary_edits(head_units,target=Fraction(1)):
    den=[3*int(np.asarray(u['gt_bins']).sum()) for u in head_units]
    if not den or len(set(den))!=1 or len({u['group'] for u in head_units})!=len(head_units):
        return dict(available=False,reason='simple event-count necessary bound needs one equal-GT clip per group')
    pooled=sum(den);required=-((-target.numerator*pooled)//(100*target.denominator))
    return dict(available=True,pooled_denominator=pooled,target_percent=[target.numerator,target.denominator],
        required_integer_gain=required,necessary_member_event_edits=(required+29)//30,
        caution='Member event modifications, not distinct bees; each deletion belongs to exactly30 windows.')

def selected_edge_audit(unit,selected,t,K_native):
    """Legal official same-direction edges, not a claim of mandatory FP/TP."""
    result=[]
    by_query={query_key(q):(m,q) for m,q in queries(unit)}
    for key in selected:
        m,q=by_query[key];native=K_native[str(m['member'])];evidence=native['evidence']
        pred={int(p[0]):int(p[1]) for p in evidence['predictions']};pid=integer(q['canonical_id'])
        signed=integer(q['event']['frame'])*integer(q['event']['direction'])
        if pred.get(pid)!=signed:raise ValueError('query does not identify the original official event')
        edges={float_thr:sorted({int(gid) for p,gid in evidence[edgekey] if int(p)==pid})
            for float_thr,edgekey in (('0.5','edges05'),('0.3','edges03'))}
        result.append(dict(query_id=key,member=str(m['member']),canonical_id=pid,event=q['event'],legal_GT_neighbors=edges,
            has_any_legal_edge={thr:bool(v) for thr,v in edges.items()},
            interpretation='Zero edges implies cannot match;nonzero edges does not imply chosen by maximum matching.'))
    return dict(selected_count=len(result),records=result,
        denominator=len(result),zero_edge_count={thr:sum(not r['has_any_legal_edge'][thr] for r in result) for thr in THRESHOLDS},
        zero_edge_fraction={thr:sum(not r['has_any_legal_edge'][thr] for r in result)/len(result) if result else None for thr in THRESHOLDS})

def tp_table(metrics):
    return {t:dict(TP=int(metrics['iou'][t]['tp']),FP=int(metrics['iou'][t]['pred'])-int(metrics['iou'][t]['tp']),
        FN=int(metrics['iou'][t]['gt'])-int(metrics['iou'][t]['tp'])) for t in THRESHOLDS}

def compact_native(native):
    return dict(nG=native['nG'],nP=native['nP'],pred_bins=native['pred_bins'],gt_bins=native['gt_bins'],W=native.get('W'),
        iou={t:{k:native['iou'][t][k] for k in ('tp','pred','gt','f1')} for t in THRESHOLDS})

def materialize_member(member,selected,t,nf):
    """Append certified actual rows only; native IDs/order and every old run stay."""
    import queries as query_core
    raw=np.asarray(member['rows'],float).reshape(-1,7);sources=list(map(tuple,member['sources']))
    before=query_core.prepare(raw,sources,nf);table={query_key(q):q for q in member['queries']}
    if len(selected)!=len(set(selected)) or not set(selected)<=table.keys():raise ValueError('foreign/repeated member query')
    added=[];keys=[];provenance=[]
    for key in sorted(selected):
        q=table[key];c=candidate(q,t)
        if c is None:raise ValueError('unconfirmed selection')
        row=np.asarray(c['added_row'],float)
        proof=query_core.insertion_safety(raw,sources,q,row,c['source_key'],nf)
        if not proof['safe']:raise ValueError('individual native safety changed '+proof['reason'])
        added.append(row);keys.append(tuple(c['source_key']));provenance.append(dict(query_id=key,gt_row_key=c['GT_row_key'],source_key=c['source_key']))
    rows=np.vstack((raw,np.asarray(added,float).reshape(-1,7)));all_sources=sources+keys
    after=query_core.prepare(rows,all_sources,nf)
    if any(after['raw_to_canonical'][i]!=alias for i,alias in before['raw_to_canonical'].items()):raise ValueError('joint insertion changes original run alias')
    for j,key in enumerate(sorted(selected)):
        if after['raw_to_canonical'][len(raw)+j]!=table[key]['canonical_id']:raise ValueError('joint addition is not in its fixed run')
    if rows[:len(raw)].tobytes()!=raw.tobytes():raise AssertionError('original K changed')
    return rows,dict(sources=[list(k) for k in all_sources],provenance=provenance,original_K_exact=True,all_original_run_aliases_exact=True)

def _common():
    # Published, already verified low-level seal/serialization helpers only.
    from tools.study203 import score203
    return score203.common

class NativeCache:
    """One official score per FULL geometry/ID/order/GT/clock/polygon state.

    Different masks producing exactly the same rows share the score. Every
    request still carries its own materialization/provenance hash. Historical
    scores may be reused only after caller verifies their sealed leaf and all
    exact input/scorer hashes; they are counted separately from fresh matches.
    """
    def __init__(self,out,scorer,writer,code_identity,historical=None):
        self.out=Path(out);self.scorer=scorer;self.writer=writer;self.code=code_identity;self.historical=historical
        self.seen=set();self.calls=[];self.fresh=self.historical_used=self.resumed=0

    def score(self,unit,member,rows,proof,gt,arm):
        common=_common();rows=np.asarray(rows,float);raw=np.asarray(member['rows'],float);gt=np.asarray(gt,float)
        if rows.ndim!=2 or rows.shape[1]!=7 or rows[:len(raw)].tobytes()!=raw.tobytes():raise ValueError('complete native K prefix differs')
        identity=dict(unit=unit['unit'],member=str(member['member']),rows_sha256=array_sha(rows),source_map_sha256=object_sha(proof['sources']),GT_sha256=array_sha(gt),
            polygon_sha256=array_sha(unit['polygon']),nf=integer(unit['nf']),scorer_code=self.code)
        token=object_sha(identity);path=self.out/token;shared=token in self.seen
        if path.exists():
            receipt=common.verify(path)
            if receipt['identity']!=identity:raise ValueError('resumed geometry identity differs')
            reread=np.loadtxt(path/'mot.txt',delimiter=',',usecols=range(7),ndmin=2) if len(rows) else np.empty((0,7))
            if reread.tobytes()!=rows.tobytes():raise ValueError('saved MOT float roundtrip differs')
            native=json.loads((path/'score.json').read_text());origin=receipt['origin']
            if not shared:self.resumed+=1
        else:
            old=self.historical(unit,member,identity) if self.historical and rows.tobytes()==raw.tobytes() else None
            if old is not None:
                for key in ('rows_sha256','GT_sha256','polygon_sha256','nf'):
                    if old[key]!=identity[key]:raise ValueError('historical exact geometry/GT/clock differs')
                if old['scorer_code']!=self.code:raise ValueError('historical official scorer differs')
                native=old['score'];origin='historical';history={k:v for k,v in old.items() if k!='score'};self.historical_used+=1
            else:native=self.scorer(gt,rows,unit['polygon'],unit['nf']);origin='fresh';history=None;self.fresh+=1
            path.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+token+'-',dir=path.parent))
            try:
                written=self.writer(rows,temp/'mot.txt');reread=np.loadtxt(temp/'mot.txt',delimiter=',',usecols=range(7),ndmin=2) if len(rows) else np.empty((0,7))
                if reread.tobytes()!=rows.tobytes():raise ValueError('physical MOT serialization differs')
                common.dump(temp/'score.json',native);common.dump(temp/'receipt.json',dict(identity=identity,origin=origin,historical=history,writer=written))
                common.seal(temp);temp.rename(path)
            finally:
                if temp.exists():shutil.rmtree(temp)
        self.seen.add(token)
        self.calls.append(dict(unit=unit['unit'],member=str(member['member']),arm=arm,path=token,within_job_shared=shared,
            origin=origin,provenance_sha256=object_sha(proof),physical_seal=(path/'SEALED.sha256').read_text().strip()))
        return native

def score_unit(unit,selections,joint_validate,gt,cache):
    """All joint checks pass BEFORE any state is scored. No fixed event graph F1."""
    for arm,t in (('R05','0.5'),('R03','0.3')):
        audit=joint_validate(unit,selections[arm],t)
        if not audit['safe']:raise ValueError('final joint witness is invalid '+str(audit))
    result={};K_native={}
    for arm,t in (('K','0.5'),('R05','0.5'),('R03','0.3')):
        members=[];wanted=set(selections.get(arm,[]))
        for m in unit['members']:
            selected=sorted(wanted & {query_key(q) for q in m['queries']})
            rows,proof=materialize_member(m,selected,t,unit['nf'])
            native=cache.score(unit,m,rows,proof,gt,arm)
            expected=np.asarray(m['baseline_windows'],np.int64).copy()
            for q in m['queries']:
                if query_key(q) in selected:expected-=event_column(q,unit['nf'])
            if not np.array_equal(native['pred_bins'],expected):raise ValueError('physical native events disagree with predicted query deletion')
            if not np.array_equal(native['gt_bins'],unit['gt_bins']):raise ValueError('physical native GT clock differs from owner counts')
            if arm=='K':K_native[str(m['member'])]=native
            members.append(dict(member=str(m['member']),metrics=compact_native(native),selected_query_ids=selected,TP_FP_FN=tp_table(native)))
        summed=sum(np.asarray(m['metrics']['pred_bins']) for m in members)
        result[arm]=dict(unit=unit['unit'],domain=unit['domain'],group=unit['group'],video=unit['video'],members=members,
            numerator=int(np.abs(summed-3*np.asarray(unit['gt_bins'])).sum()),denominator=3*int(np.asarray(unit['gt_bins']).sum()))
    result['selected_original_event_edges']={arm:selected_edge_audit(unit,selections[arm],t,K_native)
        for arm,t in (('R05','0.5'),('R03','0.3'))}
    return result

def summarize_physical(scored):
    from tools.study198.run198 import combine
    domains={}
    for h in sorted({v['domain'] for v in scored['K']}):
        methods={}
        for arm in ARMS:
            rows=[v for v in scored[arm] if v['domain']==h];groups=[]
            E1_groups=[]
            for group in sorted({v['group'] for v in rows}):
                vv=[v for v in rows if v['group']==group];combined=combine(vv);W=[]
                for i,m in enumerate(vv[0]['members']):
                    numerator=sum(int(np.abs(np.asarray(v['members'][i]['metrics']['pred_bins'])-np.asarray(v['members'][i]['metrics']['gt_bins'])).sum()) for v in vv)
                    denominator=sum(30*int(v['members'][i]['metrics']['nG']) for v in vv)
                    if not denominator:raise ValueError('undefined zeroGT whole member/group')
                    fraction=Fraction(100*numerator,denominator);W.append(fraction)
                E1=sum(W,Fraction())/3;E1_groups.append(E1)
                combined.update(W_mean_member=float(E1),W_mean_member_fraction=[E1.numerator,E1.denominator],
                    TP_FP_FN={t:[dict(member=m['member'],TP=m['tp'],FP=m['pred']-m['tp'],FN=m['gt']-m['tp']) for m in combined['iou'][t]['members']] for t in THRESHOLDS})
                groups.append(combined)
            if any(g['denominator']<=0 for g in groups):raise ValueError('undefined zero-GT whole group')
            E3=sum((Fraction(100*g['numerator'],g['denominator']) for g in groups),Fraction())/len(groups)
            f1={t:sum((sum((Fraction(2*m['tp'],max(m['pred']+m['gt'],1)) for m in g['iou'][t]['members']),Fraction())/3 for g in groups),Fraction())/len(groups) for t in THRESHOLDS}
            E1=sum(E1_groups,Fraction())/len(E1_groups)
            methods[arm]=dict(groups=groups,E3=float(E3),E3_fraction=[E3.numerator,E3.denominator],W_mean_member=float(E1),W_mean_member_fraction=[E1.numerator,E1.denominator],F1={t:float(v) for t,v in f1.items()},F1_fraction={t:[v.numerator,v.denominator] for t,v in f1.items()})
        domains[h]=dict(methods=methods)
    return domains

def synthetic_native():
    """Tiny complete-scene official check; run when this worker owns CPU."""
    import queries as query_core
    from tools.study195.run import score_member_native
    from tools.study195.freeze import write_mot_exact
    def row(f,p,x,y=0):return [f,p,x,y,2,2,1]
    polygon=[[0,-10],[50,-10],[50,50],[0,50]];nf=7
    raw=np.asarray([row(2,1,-4),row(3,1,-3),row(4,1,1),row(1,2,-4,10),row(2,2,1,10)],float)
    ret=np.asarray(row(5,1,-3),float);gt=np.vstack((raw,ret))[:,:6]
    gtcounts=bins([(2,1)],nf);members=[]
    for slot in range(3):
        sources=[('synthetic'+str(slot),i) for i in range(len(raw))];roster=query_core.generate(raw,sources,polygon,nf,'synthetic',slot)
        qq=[]
        for q in roster['queries']:
            yes=q['native_id']==1;c=dict(query_id=q['query_id'],frame=5,added_row=ret.tolist(),GT_row_key=['synthetic',1,0,5,5],
                source_key=['GTsynthetic',5],event_delete_only=True,delta_windows=(-event_column(q,nf)).tolist()) if yes else None
            q['labels']={t:dict(confirmed=yes,candidate=c,GT_run_key=['synthetic',1,0]) for t in THRESHOLDS};qq.append(q)
        members.append(dict(member=str(slot+10),member_index=slot,rows=raw,sources=sources,queries=qq,
            baseline_windows=bins([(2,1),(4,1)],nf).tolist()))
    u=dict(unit='synthetic',domain='35',group=2,video='synthetic',nf=nf,polygon=polygon,members=members,
        pred_sum=(3*np.asarray(members[0]['baseline_windows'])).tolist(),gt_bins=gtcounts.tolist())
    valid=lambda *args:dict(safe=True,reason='synthetic fixed source')
    greedy=greedy_group([u],'0.5',valid)
    assert greedy['selected_query_count']==3 and greedy['selected_GT_runs']==1 and greedy['gain_numerator']==90
    selections=dict(K=[],R05=greedy['selected']['synthetic'],R03=greedy['selected']['synthetic'])
    with tempfile.TemporaryDirectory() as temp:
        cache=NativeCache(Path(temp)/'physical',score_member_native,write_mot_exact,dict(synthetic='official native'))
        from stage206 import prefix_scorer
        u['_actual_gt']=gt
        checked=greedy_group([u],'0.5',valid,on_accept=prefix_scorer([u],cache))
        assert [s['physical_prefix']['group_numerator'] for s in checked['steps']]==[60,30,0]
        scored=score_unit(u,selections,valid,gt,cache)
        assert cache.fresh==6 and len(cache.seen)==6 and len(cache.calls)==18
        assert scored['K']['numerator']==90 and scored['R05']['numerator']==scored['R03']['numerator']==0
        assert scored['selected_original_event_edges']['R05']['zero_edge_count']=={'0.5':3,'0.3':3}
        for m in scored['R05']['members']:assert m['TP_FP_FN']['0.5']==dict(TP=1,FP=0,FN=0)
        before=len(cache.calls);score_unit(u,selections,valid,gt,cache)
        assert cache.fresh==6 and len(cache.calls)==before+9
        # Existence of some legal edge is NOT proof the selected event matches.
        q=next(q for q in members[0]['queries'] if q['native_id']==1)
        fake={str(m['member']):dict(evidence=dict(predictions=[[1,4]],edges05=[[1,99]],edges03=[])) for m in members}
        audit=selected_edge_audit(u,[q['query_id']],'0.5',fake)
        assert audit['zero_edge_count']=={'0.5':0,'0.3':1}
    print('PASS native synthetic complete MOT at every prefix+final;3 modifications/1GT-run;fresh6;TPFPFN;0-vs-some legal edges')

def selftest():
    # A return can harm an under-count window; the all-query relaxation is an
    # upper bound, not a promise that deleting every confirmed event helps.
    def query(key,f,d,bee):
        q=dict(query_id=key,canonical_id=int(key),event=dict(frame=f,direction=d))
        c=dict(query_id=key,frame=f+1,added_row=[f+1,int(key),0,0,2,2,1],GT_row_key=['video',bee,f+1],event_delete_only=True,
            delta_windows=(-event_column(q,751)).tolist())
        q['labels']={t:dict(confirmed=True,candidate=c,GT_run_key=['video',bee]) for t in THRESHOLDS};return q
    qq=[query('1',25,1,1),query('2',26,1,2),query('3',751,-1,3)]
    gt=bins([(25,1),(751,-1)],751);pred=3*gt;pred[0,0]+=1
    u=dict(unit='toy',domain='35',group=2,video='video',nf=751,pred_sum=pred.tolist(),gt_bins=gt.tolist(),
        members=[dict(member='oldmember',queries=qq)])
    b=bound_unit(u,'0.5');greedy=greedy_group([u],'0.5',lambda *args:dict(safe=True,reason='synthetic'))
    assert greedy['gain_numerator']<=b['gain_upper_numerator'] and not greedy['steps']
    # Same-second two candidates eachhelp30 alone; only one fits residual1.
    gt=bins([(25,1)],751);pred=3*gt+event_column(qq[0],751)
    u=dict(u,pred_sum=pred.tolist(),gt_bins=gt.tolist());g=greedy_group([u],'0.5',lambda *args:dict(safe=True,reason='synthetic'))
    assert g['selected_query_count']==1 and g['steps'][0]['query_id']=='1' and g['gain_numerator']==30
    denied=greedy_group([u],'0.5',lambda *args:dict(safe=False,reason='joint conflict'))
    assert not denied['steps'] and denied['rejected']
    from tools.study103.swap_score import counts_from_events
    for nf in (1,24,25,26,749,750,751):
        es=[(1,1),(nf,-1)];assert np.array_equal(bins(es,nf),counts_from_events(np.asarray(es,int),nf))
    # Fixed35 denominators:4*(90*129)=46440,1pp needs465 units/16events.
    necessary=necessary_edits([dict(u,group=i,gt_bins=bins([(1,1)]*129,751).tolist()) for i in range(4)])
    assert (necessary['pooled_denominator'],necessary['required_integer_gain'],necessary['necessary_member_event_edits'])==(46440,465,16)
    print('PASS exact30partial windows; per-cell optimistic U; stable positive greedy; joint reject; fixed35 465/16 necessary bound')

if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    for name in ('synthetic','synthetic-native','score'):mode.add_argument('--'+name,action='store_true')
    p.add_argument('--root',type=Path);p.add_argument('--prereg');p.add_argument('--frozen-seal');p.add_argument('--owners-seal')
    p.add_argument('--source203-score',type=Path);p.add_argument('--source203-score-seal');p.add_argument('--owners-released',action='store_true');p.add_argument('--field-go',action='store_true')
    a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
    if a.synthetic:selftest()
    elif a.synthetic_native:synthetic_native()
    else:
        if not all((a.root,a.prereg,a.frozen_seal,a.owners_seal,a.source203_score,a.source203_score_seal,a.owners_released,a.field_go)):
            p.error('score requires corrected pinned owner/frozen seals, published prereg, historical203 seal, owner CPU release, and root GO')
        from stage206 import field
        field(a)
