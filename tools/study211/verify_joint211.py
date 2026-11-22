"""Independent joint211 audit primitives. SOURCE/SYNTHETIC until CPU grant.

Own materialization, original-run constraints, legal subsets, policy/dose ties,
integer windows and Pareto/gate arithmetic. The field adapter will explicitly
declare reused audit211/native event legality and ordered-row Hungarian cache.
No producer joint merger/selector/model fitter is imported by these primitives.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from fractions import Fraction as F
from itertools import combinations
from numbers import Integral
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np

POLICIES=('K','Q','L','X','V','C','Z')
def integer(x):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral):raise ValueError('integer required')
    return int(x)
def family(a):
    f=a['family']
    if f in ('C','209'):return 'C'
    if f in ('I','210'):return 'I'
    raise ValueError('unrecognized frozen action family')
def endpoints(a):
    p=tuple(map(integer,a['canonical_ids']))
    if len(p)!=2 or not 0<p[0]<p[1]:raise ValueError('immutable ascending original run pair required')
    return p
def action_key(a):return (family(a),*endpoints(a))
def state_key(s,actions):return tuple(sorted(action_key(actions[k]) for k in s['selected_action_ids']))
def array_sha(a):
    a=np.ascontiguousarray(a,dtype='<f8')
    return hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n'+a.tobytes()).hexdigest()

def original_runs(rows,canon):
    """Second implementation of native ID+100000*official >25 gap ordinal."""
    R=np.asarray(rows,np.float64);C=np.asarray(canon)
    if R.ndim!=2 or R.shape[1]!=7 or not np.isfinite(R).all() or (R[:,4:6]<=0).any() or not np.array_equal(R[:,:2],np.rint(R[:,:2])) or (R[:,:2]<1).any():raise ValueError('invalid original native rows')
    if C.shape!=(len(R),) or not np.issubdtype(C.dtype,np.integer):raise ValueError('original canonical map required')
    expected=np.empty(len(R),np.int64);runs={}
    for native_id in sorted(set(map(int,R[:,1]))):
        indices=sorted(np.flatnonzero(R[:,1]==native_id).tolist(),key=lambda i:(R[i,0],i));blocks=[[]]
        for i in indices:
            if blocks[-1] and R[i,0]<=R[blocks[-1][-1],0]:raise ValueError('duplicate native track/date')
            if blocks[-1] and R[i,0]-R[blocks[-1][-1],0]>25:blocks.append([])
            blocks[-1].append(i)
        for ordinal,block in enumerate(blocks):
            cid=native_id+100000*ordinal
            if cid in runs or cid>=2**53:raise ValueError('original official alias collision')
            runs[cid]=block;expected[block]=cid
    if not np.array_equal(C,expected):raise ValueError('original canonical map differs official gap runs')
    return R,C.astype(np.int64,copy=False),runs

def legal_subsets(actions):
    """Independent combinations enumeration of all cross-family matchings."""
    keys=sorted(actions,key=lambda k:action_key(actions[k]))
    if len(keys)>20:raise ValueError('small frozen joint graph expected')
    answer=[]
    for n in range(len(keys)+1):
        for selected in combinations(keys,n):
            nodes=[r for k in selected for r in endpoints(actions[k])]
            if len(nodes)==len(set(nodes)):answer.append(tuple(selected))
    return answer

def merge(rows,canon,actions,selected):
    R,C,runs=original_runs(rows,canon);ids=list(selected)
    if len(ids)!=len(set(ids)) or not set(ids)<=actions.keys():raise ValueError('foreign/repeated action')
    if not ids:return R.copy(),dict(output_raw_indices=list(range(len(R))),deleted_raw_indices=[],actions=[])
    output=R.copy();output[:,1]=C;lost=set();used=set();proof=[]
    for key in sorted(ids,key=lambda k:action_key(actions[k])):
        action=actions[key];a,b=endpoints(action)
        if a not in runs or b not in runs or used&{a,b}:raise ValueError('shared/foreign original run')
        A={int(R[i,0]):i for i in runs[a]};B={int(R[i,0]):i for i in runs[b]};dates=sorted(A.keys()&B.keys());winners=[];deleted=[]
        if family(action)=='C':
            if not dates:raise ValueError('co-observed family lacks common original date')
            for frame in dates:
                contenders=(A[frame],B[frame]);winner=min(contenders,key=lambda i:(-float(R[i,6]),i));loser=next(i for i in contenders if i!=winner)
                lost.add(loser);deleted.append(loser);winners.append(winner)
            source=action.get('source_pair')
            if source is not None and (dates!=source['shared_dates'] or [[A[f],B[f]] for f in dates]!=source['row_indices']):raise ValueError('frozen co-observation provenance differs')
        elif dates:raise ValueError('interleave family shares an actual date')
        allindices=runs[a]+runs[b];retained=sorted((i for i in allindices if i not in lost),key=lambda i:R[i,0]);clock=[int(R[i,0]) for i in retained]
        if any(not 0<y-x<=25 for x,y in zip(clock,clock[1:])):raise ValueError('merged whole run splits or repeats a date')
        output[allindices,1]=a;used.update((a,b));proof.append(dict(action_id=key,canonical_ids=[a,b],shared_dates=dates,winner_raw_indices=winners,deleted_raw_indices=deleted))
    order=sorted(set(range(len(R)))-lost,key=lambda i:(R[i,0],i));physical=output[order]
    if physical[:,[0,2,3,4,5,6]].tobytes()!=R[order][:,[0,2,3,4,5,6]].tobytes() or len(set(map(tuple,physical[:,:2])))!=len(physical):raise AssertionError('raw observations or unique physical track/date changed')
    return physical,dict(output_raw_indices=order,deleted_raw_indices=sorted(lost),actions=proof)

def clock(events,nf):
    E=np.asarray(list(events),int).reshape(-1,2);parts=[]
    if len(E) and ((E[:,0]<1).any() or (E[:,0]>nf).any() or not np.isin(E[:,1],[-1,1]).all()):raise ValueError('invalid full-clip event clock')
    for shift in range(0,750,25):
        n=(nf+shift+749)//750;bins=(E[:,0]-1+shift)//750
        parts.append(np.bincount(2*bins+(E[:,1]<0),minlength=2*n).reshape(n,2))
    return np.vstack(parts)

def event_provenance(rows,polygon,output_raw_indices=None,original_canon=None):
    """Second event implementation: original-clock gap runs and side changes.

    Only the official polygon center/edge kernel is shared. The final direction
    is first-versus-last side and the timestamp is its LAST same-direction
    crossing. GT references may retain the official nonpositive extents.
    """
    from beetrack.evaluate import inside
    R=np.asarray(rows,float);bits=inside(R,polygon);events={};details={}
    order=None if output_raw_indices is None else np.asarray(output_raw_indices,int)
    parent=None if original_canon is None else np.asarray(original_canon,int)
    for pid in sorted(set(map(int,R[:,1]))):
        ix=sorted(np.flatnonzero(R[:,1]==pid).tolist(),key=lambda j:(R[j,0],j));blocks=[[]]
        for j in ix:
            if blocks[-1] and R[j,0]-R[blocks[-1][-1],0]>25:blocks.append([])
            if blocks[-1] and R[j,0]<=R[blocks[-1][-1],0]:raise ValueError('event track has duplicate date')
            blocks[-1].append(j)
        for ordinal,block in enumerate(blocks):
            cid=pid+100000*ordinal
            if bits[block[0]]==bits[block[-1]]:continue
            direction=1 if bits[block[-1]] else -1;cross=[]
            for old,new in zip(block,block[1:]):
                if bits[old]!=bits[new]:cross.append((old,new,1 if bits[new] else -1))
            old,new,_=next(c for c in reversed(cross) if c[2]==direction)
            frame=int(R[new,0])
            if cid in events:raise ValueError('canonical event alias collision')
            events[cid]=(frame,direction);info=dict(id=cid,frame=frame,direction=direction,signed_frame=frame*direction)
            if order is not None:
                info['selected_crossing_raw_index']=int(order[new]);info['crossings']=[dict(before_raw=int(order[a]),after_raw=int(order[b]),frame=int(R[b,0]),direction=d) for a,b,d in cross]
                if parent is not None:info['original_parent_ids']=sorted(set(map(int,parent[order[block]])))
            details[cid]=info
    return events,details

def matching(left,right,edges,removed=None):
    """Independent augmenting DFS; no producer matching is consulted."""
    neighbors={p:[] for p in left if p!=removed};right=set(right)
    for p,g in sorted(set(map(tuple,edges))):
        if p not in set(left) or g not in right:raise ValueError('foreign event edge')
        if p in neighbors:neighbors[p].append(g)
    matched={}
    def visit(p,seen):
        for g in neighbors[p]:
            if g in seen:continue
            seen.add(g)
            if g not in matched or visit(matched[g],seen):matched[g]=p;return True
        return False
    for p in sorted(neighbors):visit(p,set())
    return len(matched)

def peer_summary(graphs):
    if len(graphs)!=2:raise ValueError('two fixed peer references required')
    counts=[];js=[];ms=[];left=None
    for g in graphs:
        current=tuple(sorted(g['left']))
        if left is not None and current!=left:raise ValueError('state event vertices differ by peer')
        left=current;J=len({p for p,r in g['edges']});M=matching(g['left'],g['right'],g['edges'])
        counts.append(int(np.abs(np.asarray(g['pred_windows'])-np.asarray(g['reference_windows'])).sum()));js.append(J);ms.append(M)
    return dict(nP=len(left),meanJ=sum(js)/2,meanM=sum(ms)/2,meanCountL1=sum(counts)/2)

def utilities(state,K,model):
    q=model['Q']['a'] if isinstance(model['Q'],dict) else model['Q'];L=model['L'];X=model['X']
    dn=state['nP']-K['nP'];dJ=state['meanJ']-K['meanJ'];dM=state['meanM']-K['meanM']
    return dict(Q=(2*q-1)*dn,L=(2*L['a']-1)*dn+2*L['b']*dJ,
        X=(2*X['a']-1)*dn+2*X['b']*dM,V=2*dM-dn,C=K['meanCountL1']-state['meanCountL1'])

def choose(states,actions,model):
    """Frozen utilities and EXACT family-dose geometry control; no fit."""
    if len({s['state_id'] for s in states})!=len(states):raise ValueError('duplicate state ID')
    subsets=[tuple(sorted(s['selected_action_ids'])) for s in states]
    if len(set(subsets))!=len(subsets) or set(subsets)!={tuple(sorted(s)) for s in legal_subsets(actions)}:raise ValueError('joint legal state family is incomplete/extra')
    K=next(s for s in states if not s['selected_action_ids']);U={s['state_id']:utilities(s,K,model) for s in states};chosen={'K':K['state_id']}
    tie=lambda s:(len(s['selected_action_ids']),state_key(s,actions))
    for policy in ('Q','L','X','V','C'):chosen[policy]=min(states,key=lambda s:(-round(U[s['state_id']][policy],10),*tie(s)))['state_id']
    ranks={}
    for f in ('C','I'):
        subset=[key for key in actions if family(actions[key])==f]
        metric='mean_shared_iou' if f=='C' else 'min_local_norm'
        for key in subset:
            if not math.isfinite(actions[key][metric]):raise ValueError('nonfinite frozen geometry rank')
        subset.sort(key=lambda key:((-1 if f=='C' else 1)*actions[key][metric],action_key(actions[key])))
        for n,key in enumerate(subset):ranks[key]=F(n,max(1,len(subset)-1))
    x=next(s for s in states if s['state_id']==chosen['X'])
    dose=lambda s:tuple(sum(family(actions[key])==f for key in s['selected_action_ids']) for f in ('C','I'))
    target=dose(x);eligible=[s for s in states if dose(s)==target]
    z=min(eligible,key=lambda s:(sum((ranks[key] for key in s['selected_action_ids']),F()),state_key(s,actions)));chosen['Z']=z['state_id']
    rank=sum((ranks[key] for key in z['selected_action_ids']),F());pure={};family_only={}
    for f in ('C','I'):
        eligible=[s for s in states if all(family(actions[key])==f for key in s['selected_action_ids'])]
        scoped={'K':K['state_id']}
        for policy in ('Q','L','X','V','C'):
            scoped[policy]=min(eligible,key=lambda s:(-round(U[s['state_id']][policy],10),*tie(s)))['state_id']
        pure[f]=scoped['X']
        sx=next(s for s in eligible if s['state_id']==scoped['X']);target_f=dose(sx)
        candidates=[s for s in eligible if dose(s)==target_f]
        sz=min(candidates,key=lambda s:(sum((ranks[key] for key in s['selected_action_ids']),F()),state_key(s,actions)))
        scoped['Z']=sz['state_id'];sr=sum((ranks[key] for key in sz['selected_action_ids']),F())
        family_only[f]=dict(choices=scoped,legal_state_ids=[s['state_id'] for s in eligible],
            Z_dose=target_f,Z_rank_fraction=[sr.numerator,sr.denominator])
    contrast={s['state_id']:(s['meanJ']-K['meanJ'])-(s['meanM']-K['meanM']) for s in states}
    return dict(choices=chosen,utilities=U,Z_dose=target,Z_rank_fraction=[rank.numerator,rank.denominator],pure_X=pure,family_only=family_only,
        peer_J_minus_M_deltas=contrast,observed_peer_assignment_contrast=any(v!=0 for v in contrast.values()))

def f1(tp,pred,gt):return F(2*int(tp),max(int(pred)+int(gt),1))
def pareto(states,t):
    """W numerator minimized, ACTUAL F1 maximized. E is reported separately."""
    def quality(s):
        m=s['iou'][t];return f1(m['tp'],m['pred'],m['gt'])
    return [s['state_id'] for s in states if not any((u['numerator']<=s['numerator'] and quality(u)>=quality(s)) and
        (u['numerator']<s['numerator'] or quality(u)>quality(s)) for u in states)]

def strict_duplicate(before,after,old_events,new_events):
    conditions=dict(two_original_events=len(old_events)==2,one_merged_event=len(new_events)==1,
        same_direction=len({e['direction'] for e in old_events+new_events})==1)
    neighbors=lambda graph,p:{g for q,g in graph['edges'] if q==p}
    old=[neighbors(before,e['id']) for e in old_events];common=old[0] if len(old)==2 and old[0]==old[1] else set()
    conditions['same_singleton_GT_neighbor']=len(common)==1
    conditions['merged_preserves_edge']=len(new_events)==1 and bool(common & neighbors(after,new_events[0]['id']))
    b=matching(before['left'],before['right'],before['edges']);a=matching(after['left'],after['right'],after['edges'])
    conditions.update(whole_scene_TP_not_decreased=a>=b,prediction_count_minus_one=len(after['left'])==len(before['left'])-1)
    return dict(conditions=conditions,qualifies=all(conditions.values()),before_TP=b,after_TP=a,TP_unchanged=a==b,true_bee_identity_claim=False)

def selftest():
    R=np.asarray([[2,2,8,2,2,2,.8],[1,1,11,2,2,2,.7],[2,1,8,2,2,2,.8],
        [3,3,7,2,2,2,.9],[1,4,11,8,2,2,.9],[2,5,8,8,2,2,.9]],float);canon=R[:,1].astype(np.int64)
    A={'c':dict(family='C',canonical_ids=[1,2],mean_shared_iou=.9),
       'i':dict(family='I',canonical_ids=[2,3],min_local_norm=.1),
       'j':dict(family='I',canonical_ids=[4,5],min_local_norm=.2)}
    subsets=legal_subsets(A);assert len(subsets)==6 and not any('c' in s and 'i' in s for s in subsets)
    rows,proof=merge(R,canon,A,['c','j']);assert proof['deleted_raw_indices']==[2] and proof['output_raw_indices']==[1,4,0,5,3]
    assert rows[:,1].tolist()==[1,4,1,4,3]
    assert matching([1,2],[9],[(1,9),(2,9)])==1 and matching([1,2],[9],[(1,9),(2,9)],1)==1
    actions={'c0':dict(family='C',canonical_ids=[1,2],mean_shared_iou=.9),'c1':dict(family='C',canonical_ids=[3,4],mean_shared_iou=.8),
        'i0':dict(family='I',canonical_ids=[2,5],min_local_norm=.1),'i1':dict(family='I',canonical_ids=[6,7],min_local_norm=.2)}
    states=[dict(state_id=str(i),selected_action_ids=list(s),nP=10-len(s),meanJ=10-len(s),meanM=3 if set(s)=={'c1','i1'} else 1,meanCountL1=10-len(s)) for i,s in enumerate(legal_subsets(actions))]
    result=choose(states,actions,dict(Q=dict(a=.5),L=dict(a=.5,b=0),X=dict(a=.25,b=.5)))
    x=next(s for s in states if s['state_id']==result['choices']['X']);z=next(s for s in states if s['state_id']==result['choices']['Z'])
    assert len(x['selected_action_ids'])==2 and len(z['selected_action_ids'])==2
    assert set(x['selected_action_ids'])=={'c1','i1'} and set(z['selected_action_ids'])=={'c0','i1'}
    assert result['Z_dose']==tuple(sum(family(actions[k])==f for k in x['selected_action_ids']) for f in ('C','I'))
    assert strict_duplicate(dict(left=[1,2],right=[9],edges=[(1,9),(2,9)]),dict(left=[1],right=[9],edges=[(1,9)]),[dict(id=1,direction=1),dict(id=2,direction=1)],[dict(id=1,direction=1)])['qualifies']
    print('PASS independent mixed merge/tie/lost raw;cross-family matching completeness;capacity ambiguity;exact-dose selection;strict event witness;source only')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--synthetic',action='store_true',required=True);a=p.parse_args();selftest()
