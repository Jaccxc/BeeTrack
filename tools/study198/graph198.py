"""§198 full-scene endpoint-state graph evaluator and source-only labels.

Conditional on §197's verified unchanged per-frame GT ownership and canonical
allocator. Graphs reconstruct TP/F1 and dated counts, NOT strict tp_id: the
first/last assigned GT owner is not stored in event-edge evidence.
No field files, GT loaders, or artifact paths are opened by this module.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'1')

from copy import deepcopy
from fractions import Fraction
import numpy as np

from tools.study117.tracking.score117 import FORMAT, evidence_sha256, replay_case

DOMAINS=('35','33','W','G')
ARMS197=('K','A','H','G','T','M','O')


def _graph(metrics,nframes):
    original=metrics.get('evidence',metrics)
    graph={k:deepcopy(original[k]) for k in ('gt','predictions','edges05','edges03')}
    graph.update(format=FORMAT,nframes=int(nframes))
    if original.get('nframes',nframes) != nframes:
        raise ValueError('evidence/native clip clock mismatch')
    replay_case(graph)  # public official graph/date validation and matching
    return graph


def _table(graph):
    return {int(r[0]):list(r) for r in graph['predictions']}


def build_member(bundle,K_metrics,A_metrics,nframes=None,context=None):
    """Build all-197-candidate model, never just the observable subset.

    bundle needs parent_blocks[{rank,parent_id}] and all candidates with
    candidate_id,parent_id,delta,frame,direction. Physical IDs are allocated
    2*rank+1(left/uncut),2*rank+2(right), exactly §197 physical.realize.
    K/A must be the full-scene keep/all-cut graphs; GT and fixed event nodes
    must agree. Candidate state events must be exactly one signed toggle.
    """
    nf=int(nframes if nframes is not None else bundle['nframes'])
    K,A=_graph(K_metrics,nf),_graph(A_metrics,nf)
    if K['gt'] != A['gt']:
        raise ValueError('K/A canonical GT event census differs')
    blocks=list(bundle['parent_blocks'])
    parent_ids=[int(b['parent_id']) for b in blocks]
    ranks=[int(b['rank']) for b in blocks]
    if len(set(parent_ids)) != len(blocks) or len(set(ranks)) != len(blocks) or any(r<0 for r in ranks):
        raise ValueError('canonical parent/rank mapping must be unique')
    allocated={}
    for b in blocks:
        for identity in (2*int(b['rank'])+1,2*int(b['rank'])+2):
            allocated[identity]=int(b['parent_id'])
    candidates=list(bundle['candidates'])
    if len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('candidate IDs repeat')
    byparent={int(c['parent_id']):c for c in candidates}
    if len(byparent) != len(candidates) or not set(byparent) <= set(parent_ids):
        raise ValueError('exactly one candidate per known canonical parent')
    tables={'K':_table(K),'A':_table(A)}
    for state,table in tables.items():
        if not set(table) <= set(allocated):
            raise ValueError('graph event ID absent from canonical allocated-ID map')
        if state == 'K' and any(p%2 != 1 for p in table):
            raise ValueError('K contains a split-right ID, not a full keep graph')
    def node(state,pid):
        graph=K if state == 'K' else A
        return dict(parent_id=allocated[pid],allocated_id=pid,signedframe=int(tables[state][pid][1]),
                    prediction=tables[state][pid],
                    edges={t:[int(g) for p,g in graph[key] if p == pid]
                           for t,key in (('0.5','edges05'),('0.3','edges03'))})
    fixed,options=[],[]
    for parent in parent_ids:
        nodes={s:[node(s,p) for p in tables[s] if allocated[p] == parent] for s in ('K','A')}
        if any(len(v)>1 for v in nodes.values()):
            raise ValueError('parent state has more than one event; not §197 family')
        if parent not in byparent:
            if nodes['K'] != nodes['A']:
                raise ValueError('fixed noncandidate event/edge changed between K/A')
            fixed.extend(nodes['K'])
            continue
        c=byparent[parent]
        delta=int(c['delta'])
        if delta not in (-1,1) or len(nodes['A'])-len(nodes['K']) != delta:
            raise ValueError('candidate is not exactly one added/removed event')
        present=(nodes['A'] or nodes['K'])[0]
        signed=int(c['frame'])*int(c['direction'])
        if int(c['direction']) not in (-1,1) or present['signedframe'] != signed:
            raise ValueError('candidate event clock/direction differs from graph')
        options.append(dict(candidate_id=c['candidate_id'],parent_id=parent,delta=delta,
                            frame=int(c['frame']),direction=int(c['direction']),
                            K=nodes['K'][0] if nodes['K'] else None,
                            A=nodes['A'][0] if nodes['A'] else None))
    model=dict(schema='endpoint198-full-scene-state-graph-v1',nframes=nf,gt=K['gt'],
               fixed=fixed,candidates=options,context={} if context is None else deepcopy(context),
               source_inputs={s:deepcopy(m.get('evidence',m).get('inputs_sha256')) for s,m in (('K',K_metrics),('A',A_metrics))},
               source_graph_sha256={s:evidence_sha256(m.get('evidence',m)) for s,m in (('K',K_metrics),('A',A_metrics))},
               strict_tp_id='not reconstructible from event graphs; never inferred',
               premise='197 canonical allocation and full-frame GT owner/tie invariance')
    assert_reference(model,[],K_metrics)
    assert_reference(model,[c['candidate_id'] for c in candidates],A_metrics)
    return model


def evaluate(model,cut_ids=(),*,threshold=None):
    """Full-scene official max-cardinality replay for any 197 candidate mask."""
    if threshold is not None and str(threshold) not in ('0.5','0.3'):
        raise ValueError('only official 0.5 and supplemental 0.3 supported')
    chosen=list(cut_ids);known={c['candidate_id'] for c in model['candidates']}
    if len(set(chosen)) != len(chosen) or not set(chosen) <= known:
        raise ValueError('unique known 197 candidate IDs required')
    chosen=set(chosen)
    nodes=list(model['fixed'])
    for c in model['candidates']:
        selected=c['A'] if c['candidate_id'] in chosen else c['K']
        if selected is not None:nodes.append(selected)
    nodes.sort(key=lambda n:n['allocated_id'])
    if len({n['allocated_id'] for n in nodes}) != len(nodes):
        raise ValueError('mixed physical event IDs collide')
    evidence=dict(format=FORMAT,nframes=model['nframes'],gt=deepcopy(model['gt']),
                  predictions=[list(n['prediction']) for n in nodes],
                  edges05=sorted([[n['allocated_id'],g] for n in nodes for g in n['edges']['0.5']]),
                  edges03=sorted([[n['allocated_id'],g] for n in nodes for g in n['edges']['0.3']]))
    score=replay_case(evidence)
    nG,nP=len(model['gt']),len(nodes)
    iou={t:dict(s,pred=nP,gt=nG) for t,s in score['iou'].items()}
    from tools.study103.swap_score import counts_from_events
    pred_events=[[abs(n['signedframe']),1 if n['signedframe']>0 else -1] for n in nodes]
    gt_events=[[abs(r[1]),1 if r[1]>0 else -1] for r in model['gt']]
    return dict(nG=nG,nP=nP,iou=iou,
                pred_bins=counts_from_events(np.array(pred_events,int).reshape(-1,2),model['nframes']).tolist(),
                gt_bins=counts_from_events(np.array(gt_events,int).reshape(-1,2),model['nframes']).tolist(),
                window=score['window'],W=score['w30_percent'],events=pred_events,evidence=evidence,
                strict_tp_id=model['strict_tp_id'])


def assert_reference(model,cut_ids,reference):
    """Check every reconstructible field; explicitly exclude strict tp_id."""
    value=evaluate(model,cut_ids)
    graph=reference.get('evidence',reference)
    for key in ('gt','predictions','edges05','edges03'):
        # Official archive IDs are canonical sorted; edge list order is inert.
        actual=value['evidence'][key];wanted=graph[key]
        if sorted(map(tuple,actual)) != sorted(map(tuple,wanted)):
            raise AssertionError('197 graph mismatch '+key)
    for key in ('nG','nP','pred_bins','gt_bins','window','W'):
        if key in reference and value[key] != reference[key]:
            raise AssertionError('197 numeric mismatch '+key)
    if 'iou' in reference:
        for threshold in ('0.5','0.3'):
            for key in ('tp','pred','gt','f1','matching','by_direction'):
                if key in reference['iou'][threshold] and value['iou'][threshold][key] != reference['iou'][threshold][key]:
                    raise AssertionError('197 matching mismatch '+threshold+'/'+key)
    return dict(graph=True,TP_F1_and_full_window_counts=True,strict_tp_id_checked=False)


def source_labels(model):
    """z is event-ON marginal TP, even for a cut whose action delta is -1."""
    base=evaluate(model,[])['iou']['0.5']['tp']
    labels={}
    for c in model['candidates']:
        switched=evaluate(model,[c['candidate_id']])['iou']['0.5']['tp']
        on,off=(switched,base) if c['delta'] == 1 else (base,switched)
        z=on-off
        if z not in (0,1):
            raise AssertionError('single event ON marginal must be 0/1')
        labels[c['candidate_id']]=dict(z=z,event_on_TP=on,event_off_TP=off,
                                      delta=c['delta'],parent_id=c['parent_id'])
    return labels


def labels_for_group(models):
    """Three actual members, source-only y and z, all other candidates at K."""
    if len(models) != 3 or any(m['nframes'] != models[0]['nframes'] or m['gt'] != models[0]['gt'] for m in models):
        raise ValueError('three member models of one actual clip required')
    baseline=[evaluate(m,[]) for m in models]
    truth=np.asarray(baseline[0]['gt_bins'],np.int64)
    residual=sum(np.asarray(v['pred_bins'],np.int64) for v in baseline)-3*truth
    loss=int(np.abs(residual).sum())
    labels={}
    for model,base in zip(models,baseline):
        marginal=source_labels(model)
        for c in model['candidates']:
            cid=c['candidate_id']
            if cid in labels:raise ValueError('global 197 candidate key repeats across members')
            changed=evaluate(model,[cid])
            delta=np.asarray(changed['pred_bins'],np.int64)-np.asarray(base['pred_bins'],np.int64)
            if int(np.abs(delta).sum()) != 30:raise AssertionError('not one signed count change')
            improvement=loss-int(np.abs(residual+delta).sum())
            y=Fraction(improvement,30)
            if not -1<=y<=1:raise AssertionError('singleton source y outside [-1,1]')
            labels[cid]=dict(marginal[cid],y=float(y),y_fraction=str(y),
                            improvement_numerator=improvement,baseline_E3_numerator=loss)
    return labels


def load_source_labels(roster,heldout,*,source_domains,group_loader):
    """Whitelist descriptors BEFORE any caller-supplied graph/GT loader call.

    roster is GT-free group metadata. group_loader(descriptor) must return
    exactly that source group's three models; it must not preload all-domain
    labels. Actual GT access belongs to the caller's explicit source provider.
    """
    allowed=tuple(source_domains)
    if heldout not in DOMAINS or len(allowed)!=3 or set(allowed)!=set(DOMAINS)-{heldout}:
        raise ValueError('explicit other-three-domain source allowlist required')
    labels,receipts={},[]
    seen=set()
    for descriptor in roster:
        domain=descriptor.get('domain',descriptor.get('hive'))
        if domain not in allowed:continue
        seen.add(domain)
        models=group_loader(descriptor)  # first possible label entrance
        if any(m['context'].get('domain') != domain for m in models):
            raise ValueError('source loader returned another domain')
        local=labels_for_group(models)
        if set(labels)&set(local):raise ValueError('source global keys repeated')
        labels.update(local)
        receipts.append(dict(domain=domain,unit=descriptor['unit'],
                             graph_inputs=[m['source_inputs'] for m in models],
                             graph_sha256=[m['source_graph_sha256'] for m in models]))
    if seen != set(allowed):raise ValueError('source descriptor domains incomplete')
    return dict(source_hives=list(allowed),labels=labels,receipts=receipts,
                target_label_access='heldout descriptor never passed to group_loader')


def verify_197_records(bundles,records):
    """Pure adapter for all seven archived full physical arms, no file IO."""
    grouped={}
    for record in records:
        key=(record['unit'],str(record['member']))
        arms=grouped.setdefault(key,{})
        if record['arm'] in arms:raise ValueError('duplicate 197 arm receipt')
        arms[record['arm']]=record
    proofs=[]
    for key,arms in grouped.items():
        if set(arms)!=set(ARMS197):raise ValueError('all seven original 197 arms required')
        model=build_member(bundles[key],arms['K']['proof']['metrics'],arms['A']['proof']['metrics'],
                           context=dict(unit=key[0],member=key[1]))
        for arm,record in arms.items():
            proof=assert_reference(model,record['mask'],record['proof']['metrics'])
            proofs.append(dict(unit=key[0],member=key[1],arm=arm,**proof))
    return proofs


def models_from_197(groups,records):
    """Pure all-arm numeric adapter; caller controls domain-scoped file reads.

    records may be compact entries assembled from197 native_scores+mask files:
    {unit,member,arm,mask,proof:{metrics}}. No465MB physical receipt is needed.
    This function is a diagnostic loader, NOT a source-fold GT boundary.
    """
    bystream={}
    for r in records:
        key=(r['unit'],str(r['member']))
        arms=bystream.setdefault(key,{})
        if r['arm'] in arms:raise ValueError('duplicate197 physical stream arm')
        arms[r['arm']]=r
    expected={(g['unit'],str(m['member'])) for g in groups for m in g['members']}
    if set(bystream) != expected:raise ValueError('197 group/member roster differs')
    models,checks={},[]
    for group in groups:
        current=models.setdefault(group['unit'],{})
        for member in group['members']:
            key=(group['unit'],str(member['member']));arms=bystream[key]
            if set(arms)!=set(ARMS197):raise ValueError('all7original197arm graphs required')
            context=dict(domain=group.get('domain',group.get('hive')),unit=group['unit'],
                         member=str(member['member']),group=group['group'],video=group['video'])
            model=build_member(member['bundle'],arms['K']['proof']['metrics'],arms['A']['proof']['metrics'],
                               group['nf'],context)
            current[str(member['member'])]=model
            for arm,r in arms.items():
                proof=assert_reference(model,r['mask'],r['proof']['metrics'])
                checks.append(dict(unit=key[0],member=key[1],arm=arm,**proof))
    return dict(models=models,checks=checks,checked_streams=len(expected),checked_arms=len(checks),
                scope='Archived numeric replay under197verified-owner premise; no targetGT is opened here.')


def _toy(slot=0,domain='35',necessary_add=False):
    candidates=[dict(candidate_id=f'{domain}|{slot}|{p}',parent_id=p,delta=d,frame=f,direction=s)
                for p,d,f,s in ((10,-1,10,1),(20,1,11,1),(40,-1,13,-1),(50,1,14,1))]
    bundle=dict(nframes=75,parent_blocks=[dict(rank=i,parent_id=p) for i,p in enumerate((10,20,30,40,50))],candidates=candidates)
    pred=lambda p,t:[p,t,0.,0.,2.,2.]
    gt=[[1,10],[2,-13]]+([[3,11]] if necessary_add else [])
    K=dict(gt=gt,predictions=[pred(1,10),pred(5,12),pred(7,-13)],
           edges05=[[1,1],[5,1],[7,2]],edges03=[[1,1],[5,1],[7,2]])
    A=dict(gt=gt,predictions=[pred(4,11),pred(5,12),pred(9,14)],
           edges05=[[4,3 if necessary_add else 1],[5,1]],edges03=[[4,3 if necessary_add else 1],[5,1],[9,1]])
    return build_member(bundle,K,A,context=dict(domain=domain,unit=domain,member=str(slot)))


def selfcheck():
    from itertools import combinations
    model=_toy()
    ids=[c['candidate_id'] for c in model['candidates']]
    checked=0
    for n in range(len(ids)+1):
        for selected in combinations(ids,n):
            value=evaluate(model,selected)
            expected_pred=3+sum(c['delta'] for c in model['candidates'] if c['candidate_id'] in selected)
            assert value['nP'] == expected_pred
            assert value['iou']['0.5']['tp'] == 1+int(ids[2] not in selected)
            checked+=1
    labels=source_labels(model)
    assert [labels[c]['z'] for c in ids] == [0,0,1,0]
    assert source_labels(_toy(necessary_add=True))[ids[1]]['z'] == 1
    assert any(c['A'] and c['A']['allocated_id'] == 4 and c['parent_id'] == 20 for c in model['candidates'])
    roster=[dict(domain=d,unit=d) for d in DOMAINS]
    calls=[]
    def loader(descriptor):
        calls.append(descriptor['domain'])
        return [_toy(i,descriptor['domain']) for i in range(3)]
    for heldout in DOMAINS:
        calls.clear()
        supplied=load_source_labels(roster,heldout,source_domains=[d for d in DOMAINS if d!=heldout],group_loader=loader)
        assert heldout not in calls and set(calls)==set(DOMAINS)-{heldout}
        assert all(r['z'] in (0,1) and -1<=r['y']<=1 for r in supplied['labels'].values())
    calls.clear()
    try:load_source_labels(roster,'35',source_domains=['35','W','G'],group_loader=loader)
    except ValueError:pass
    else:raise AssertionError('heldout source label access accepted')
    assert not calls
    # Reconstruct all7 old arm masks, including a hypothetical unobservable
    # drop candidate; the evaluator must not apply198eligibility filtering.
    masks={'K':[],'A':ids,'H':[ids[0]],'G':[ids[1]],'T':[ids[2]],'M':[ids[3]],'O':ids[:2]}
    references=[dict(unit='toy',member='0',arm=arm,mask=mask,
                     proof=dict(metrics=evaluate(model,mask))) for arm,mask in masks.items()]
    bundle=dict(nframes=75,parent_blocks=[dict(rank=i,parent_id=p) for i,p in enumerate((10,20,30,40,50))],
                candidates=[{k:c[k] for k in ('candidate_id','parent_id','delta','frame','direction')} for c in model['candidates']])
    rebuilt=models_from_197([dict(unit='toy',domain='35',group=1,video='toy',nf=75,members=[dict(member='0',bundle=bundle)])],references)
    assert rebuilt['checked_arms']==7
    empty=build_member(dict(parent_blocks=[],candidates=[],nframes=75),
                       dict(gt=[],predictions=[],edges05=[],edges03=[]),
                       dict(gt=[],predictions=[],edges05=[],edges03=[]))
    assert evaluate(empty)['iou']['0.5']['f1'] == 0 and evaluate(empty)['W'] is None
    print('graph198 synthetic all16 masks/shared-GT fixed competitor/add-drop ON labels/right-ID/empty/source-domain guards passed')


if __name__ == '__main__':
    selfcheck()
