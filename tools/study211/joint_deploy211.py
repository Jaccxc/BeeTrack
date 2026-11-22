"""Source-only joint211 deployment adapter; no field loader or work on import.

Native action/state construction belongs to joint_actions211, peer graphs to
runner's module, and utility coefficients/selection to fit211. This adapter
does NOT train, alter those definitions, or infer labels from target GT.
Joint Z is exact family-dose matched to joint X, NOT the main210-only Z.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

FAMILIES=('C','I')
POLICIES=('K','Q','L','X','V','C','Z')


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def action_key(action):
    family=action['family'];ids=tuple(action['canonical_ids'])
    if family not in FAMILIES or len(ids)!=2 or any(type(i) is not int for i in ids) or not 0<ids[0]<ids[1]:
        raise ValueError('C/I family and increasing positive original canonical pair required')
    return (family,*ids)


def state_key(state,actions):
    ids=state['selected_action_ids']
    if len(set(ids))!=len(ids):raise ValueError('duplicate selected action')
    used=set();keys=[]
    for aid in ids:
        key=action_key(actions[aid]);runs=set(key[1:])
        if runs&used:raise ValueError('joint state shares an original canonical run')
        used.update(runs);keys.append(key)
    return tuple(sorted(keys))


def normalized_geometry_ranks(actions):
    """Ranks are computed on each ORIGINAL active slate, before joint conflicts.

    Use exact Fractions so equal normalized sums cannot acquire float tie bias.
    C: larger original mean shared IoU first; I: smaller min local norm first.
    """
    ranks={}
    for family in FAMILIES:
        ids=[aid for aid,a in actions.items() if a['family']==family]
        metric='mean_shared_iou' if family=='C' else 'min_local_norm'
        for aid in ids:
            action_key(actions[aid])
            v=actions[aid][metric]
            if not math.isfinite(v):raise ValueError('finite original geometric score required')
        ids.sort(key=lambda aid:((-1 if family=='C' else 1)*actions[aid][metric],action_key(actions[aid])))
        for rank,aid in enumerate(ids):ranks[aid]=Fraction(rank,max(1,len(ids)-1))
    return ranks


def z_choice(states,actions,x_state):
    """Exactly match joint-X (nC,nI), minimize rank sum, then physical keys."""
    ranks=normalized_geometry_ranks(actions)
    def dose(s):
        return tuple(sum(actions[i]['family']==f for i in s['selected_action_ids']) for f in FAMILIES)
    target=dose(x_state)
    eligible=[s for s in states if dose(s)==target]
    if not eligible:raise ValueError('X dose absent from complete legal state family')
    for s in eligible:state_key(s,actions)
    chosen=min(eligible,key=lambda s:(sum((ranks[i] for i in s['selected_action_ids']),Fraction()),state_key(s,actions)))
    value=sum((ranks[i] for i in chosen['selected_action_ids']),Fraction())
    return chosen,dict(dose_C=target[0],dose_I=target[1],rank_sum=[value.numerator,value.denominator],
                       tie_rule='family,ascending original canonical i,j',
                       normalization='rank/max(1,original family active slate size-1)')


def choose_from_utilities(states,actions,utility):
    """Main211 convention: round(U,10), fewer edits, physical canonical lex."""
    if not states:raise ValueError('empty legal state family')
    values={s['state_id']:float(utility(s)) for s in states}
    if not all(math.isfinite(v) for v in values.values()):raise ValueError('finite utility required')
    return min(states,key=lambda s:(-round(values[s['state_id']],10),
                                    len(s['selected_action_ids']),state_key(s,actions)))


def choose_joint(states,actions,utilities):
    """utilities(state) is fit211's Q/L/X/V/C relative-to-K frozen utility."""
    if len({s['state_id'] for s in states})!=len(states):raise ValueError('duplicate state ID')
    keys=[state_key(s,actions) for s in states]
    if len(set(keys))!=len(keys):raise ValueError('duplicate physical joint subset')
    empty=[s for s in states if not s['selected_action_ids']]
    if len(empty)!=1:raise ValueError('exactly one keep/empty state required')
    by_id={s['state_id']:utilities(s) for s in states}
    if any(set(u)!=set(POLICIES)-{'K','Z'} for u in by_id.values()):
        raise ValueError('exact frozen Q/L/X/V/C utilities required')
    selected={'K':empty[0]}
    for policy in ('Q','L','X','V','C'):
        selected[policy]=choose_from_utilities(states,actions,lambda s:by_id[s['state_id']][policy])
    selected['Z'],zproof=z_choice(states,actions,selected['X'])
    family_choices={};family_all={}
    for family in FAMILIES:
        subset=[s for s in states if all(actions[i]['family']==family for i in s['selected_action_ids'])]
        scoped={'K':empty[0]}
        for policy in ('Q','L','X','V','C'):
            scoped[policy]=choose_from_utilities(subset,actions,lambda s:by_id[s['state_id']][policy])
        scoped['Z'],scoped_z=z_choice(subset,actions,scoped['X'])
        family_choices[family]=scoped['X']['state_id']
        family_all[family]=dict(choices={p:s['state_id'] for p,s in scoped.items()},joint_Z=scoped_z,
                               legal_state_ids=[s['state_id'] for s in subset])
    return dict(choices={p:s['state_id'] for p,s in selected.items()},utilities=by_id,
                joint_Z=zproof,frozen_X_family_only=family_choices,family_only=family_all)


def prepare_prediction(unit,*,build_joint_states,peer_state,model,utility):
    """GT-free callback boundary. Build ALL states, then graphs, then choices.

    build_joint_states(unit)->{actions:{id:action},states:[state]}; state must
    include scope_digest, deletion_digest, state_id, selected_action_ids.
    peer_state(unit,state)->numeric full peer scene evidence. Callbacks may not
    read GT. Target GT is deliberately absent from this API.
    """
    family=build_joint_states(unit);actions,states=family['actions'],family['states']
    for s in states:
        state_key(s,actions)
        for key in ('scope_digest','deletion_digest'):
            if len(s[key])!=64 or any(c not in '0123456789abcdef' for c in s[key]):
                raise ValueError('explicit full-scope and keep/delete state hashes required')
    graphs={s['state_id']:peer_state(unit,s) for s in states}
    selected=choose_joint(states,actions,lambda s:utility(model,graphs[s['state_id']],graphs[
        next(t['state_id'] for t in states if not t['selected_action_ids'])]))
    return dict(schema='joint211-gtfree-clip-v1',unit=unit['unit'],actions=actions,states=states,
                peer_graphs=graphs,model=model,**selected)


def frozen_utilities(model,state,baseline):
    """Exact main211 relative-to-K formulas supplied by fit owner; no fitting."""
    dp=state['nPred']-baseline['nPred']
    dj=state['meanJ']-baseline['meanJ'];dm=state['meanM']-baseline['meanM']
    out={}
    for policy,dvalue in (('Q',0),('L',dj),('X',dm)):
        coef=model[policy];a,b=float(coef['a']),float(coef['b'])
        if not (0<=a<=1 and 0<=b<=1 and a+b<=1+1e-12):
            raise ValueError('sealed main coefficients outside original feasible triangle')
        out[policy]=(2*a-1)*dp+2*b*dvalue
    out['V']=2*dm-dp
    out['C']=baseline['meanCountL1']-state['meanCountL1']
    return out


def matrix_sha(rows):
    """Digest includes all row values AND ordering, not merely event/bin counts."""
    import numpy as np
    a=np.asarray(rows,dtype='<f8')
    if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all():raise ValueError('finite native Nx7 required')
    return hashlib.sha256(canonical_bytes(list(a.shape))+a.tobytes(order='C')).hexdigest()


def build_native_clip(unit,roster,model,*,joint,graphs,count_windows):
    """Full GT-free native adapter for runner graph and pairhead joint roster.

    unit contains rows,canon,polygon,nframes,peer_rows(exactly two ORIGINAL
    members),unit identifier. No GT is accepted. All singleton/combined integer
    deltas must physically recount. Private sealing retains raw-row lineage;
    publication must anonymize canonical IDs and omit row/box/source identities.
    """
    import numpy as np
    if len(unit['peer_rows'])!=2:raise ValueError('two unchanged full peer scenes required')
    rows=np.asarray(unit['rows'],float);canon=np.asarray(unit['canon'])
    baseline=np.asarray(count_windows(rows))
    if baseline.ndim!=2 or baseline.shape[1]!=2 or not np.issubdtype(baseline.dtype,np.integer):
        raise ValueError('full integer 30phase window counter required')
    actions={}
    for a in roster['actions']:
        source=a['source_pair'];family={'209':'C','210':'I'}[a['family']]
        action=dict(family=family,canonical_ids=list(a['canonical_ids']),delta_windows=a['delta_windows'])
        if family=='C':action['mean_shared_iou']=source['mean_iou']
        else:action['min_local_norm']=source['min_offset_norm']
        actions[a['action_id']]=action
    states=[];evidence={};summaries={}
    for selected in joint.feasible_subsets(roster):
        physical,proof=joint.materialize(rows,canon,roster,selected)
        actual=np.asarray(count_windows(physical));expected=baseline.copy()
        for aid in selected:expected+=np.asarray(actions[aid]['delta_windows'],dtype=expected.dtype)
        if not np.array_equal(actual,expected):raise AssertionError('joint full physical/additive windows mismatch')
        state_id='K' if not selected else digest(sorted(selected))
        state=dict(state_id=state_id,selected_action_ids=list(selected),
                   scope_digest=matrix_sha(physical),deletion_digest=digest(proof['deleted_raw_indices']),
                   retained_raw_indices=proof['output_raw_indices'],deleted_raw_indices=proof['deleted_raw_indices'],
                   physical_id_vector=physical[:,1].astype(int).tolist(),windows=actual.tolist())
        states.append(state)
        peers=[graphs.peer_graph(peer,physical,unit['polygon'],unit['nframes'],deletion=not selected)
               for peer in unit['peer_rows']]
        if peers[0]['nPred']!=peers[1]['nPred']:raise AssertionError('peer references changed left event count')
        if any(g['pred_windows']!=actual.tolist() for g in peers):raise AssertionError('graph/native windows mismatch')
        mean_count=sum(sum(abs(int(p)-int(r)) for pr,rr in zip(g['pred_windows'],g['reference_windows'])
                           for p,r in zip(pr,rr)) for g in peers)/2
        summaries[state_id]=dict(nPred=peers[0]['nPred'],meanJ=sum(g['J'] for g in peers)/2,
                                 meanM=sum(g['R'] for g in peers)/2,meanCountL1=mean_count)
        # Private numeric closure: never include image paths, geometry or GT.
        for g in peers:
            for key in ('prediction_vertices','reference_vertices'):
                for vertex in g[key]:vertex.pop('box',None)
        evidence[state_id]=peers
    baseline_id=next(s['state_id'] for s in states if not s['selected_action_ids'])
    choice=choose_joint(states,actions,lambda s:frozen_utilities(model,summaries[s['state_id']],summaries[baseline_id]))
    return dict(schema='joint211-native-gtfree-clip-v1',unit=unit['unit'],actions=actions,
                states=states,peer_graphs=evidence,state_summaries=summaries,
                model=model,source_seals=roster['source_seals'],**choice)


def assert_main210_closure(prediction,main_clip,*,main_rows):
    """All original fixed210 slate IDs MUST equal main211 all-active IDs.

    Parent's prior full-graph check proves equality/disjointness. Drift is a
    lineage failure, never permission to expand/crop either action inventory.
    main_rows(state) reconstructs frozen main physical matrices, no true GT.
    """
    actions=prediction['actions']
    actual={tuple(a['canonical_ids']) for a in actions.values() if a['family']=='I'}
    expected={tuple(p['canonical_ids']) if 'canonical_ids' in p else tuple(map(int,p['pair_id'].split('|')))
              for p in main_clip['active_pairs']}
    if actual!=expected:raise AssertionError('fixed210 active slate differs from main full-active inventory')
    pure=[s for s in prediction['states'] if all(actions[i]['family']=='I' for i in s['selected_action_ids'])]
    def key_main(s):return tuple(sorted(tuple(map(int,i.split('|'))) for i in s['selected_pair_ids']))
    mtable={key_main(s):s for s in main_clip['states']}
    jtable={tuple(sorted(tuple(actions[i]['canonical_ids']) for i in s['selected_action_ids'])):s for s in pure}
    if mtable.keys()!=jtable.keys():raise AssertionError('pure210 complete state inventory differs')
    checks=[]
    graph_keys=('nPred','nReference','J','R','edges','flags','matching','pred_windows','reference_windows')
    for key,m in mtable.items():
        s=jtable[key]
        if matrix_sha(main_rows(m))!=s['scope_digest']:raise AssertionError('main/pure210 physical matrix/order mismatch')
        if s['windows']!=m['pred_windows']:raise AssertionError('main/pure210 integer window mismatch')
        for actual_graph,reference in zip(prediction['peer_graphs'][s['state_id']],m['peer_graphs']):
            assert_reference(actual_graph,{k:reference[k] for k in graph_keys})
            for vk in ('prediction_vertices','reference_vertices'):
                stripped=[{k:v for k,v in vertex.items() if k not in ('box','constituent_original_canonical_ids')}
                          for vertex in reference[vk]]
                assert_reference(actual_graph[vk],stripped)
        checks.append(dict(state_id=s['state_id'],main_state_id=m['state_id'],
                           physical_exact=True,integer_windows_exact=True,peer_graph_exact=True))
    return dict(all_original_active_ids_exact=True,states=checks)


def score_native_states(prediction,unit,gt,*,score_native,gate,expected_gate,
                        reuse=None,scorer_receipt):
    """Post-seal complete state scoring, preserving strict physical reuse keys.

    gate must be the result of verify_target_gate with root's explicit grant.
    reuse may only key matrix/GT/clock/polygon/scorer digests, NEVER windows,
    selected arm names, or legacy original graph IDs. Pure209 reordered scenes
    therefore get fresh official frame ownership whenever physical bytes differ.
    """
    import numpy as np
    if gate!=expected_gate or gate.get('schema')!='joint211-all-four-gtfree-seal-v1':
        raise PermissionError('verified complete joint prediction gate required before target GT')
    cache={} if reuse is None else reuse
    original=np.asarray(unit['rows'],float);result={}
    gt_hash=matrix_sha(gt) if np.asarray(gt).shape[1]==7 else hashlib.sha256(np.asarray(gt,dtype='<f8').tobytes()).hexdigest()
    for state in prediction['states']:
        selected=np.asarray(state['retained_raw_indices'],dtype=int)
        physical=original[selected].copy()
        physical[:,1]=state['physical_id_vector']
        if matrix_sha(physical)!=state['scope_digest']:raise AssertionError('sealed scope reconstruction mismatch')
        if digest(state['deleted_raw_indices'])!=state['deletion_digest']:raise AssertionError('sealed deletion mismatch')
        key=digest(dict(physical=state['scope_digest'],gt=gt_hash,nframes=unit['nframes'],
                        polygon=np.asarray(unit['polygon']).tolist(),scorer=scorer_receipt))
        reused=key in cache
        if not reused:cache[key]=score_native(gt,physical,unit['polygon'],unit['nframes'])
        metric=cache[key]
        result[state['state_id']]=dict(metrics=metric,scope_digest=state['scope_digest'],
                                      deletion_digest=state['deletion_digest'],strict_reuse_key=key,reused=reused)
    return result


def write_four_fold_seal(directory,documents,receipts):
    """Write once only, after all four GT-free prediction documents exist."""
    if set(documents)!={'35','33','W','G'}:raise ValueError('complete four-fold deployment required')
    directory=Path(directory)
    if directory.exists():raise FileExistsError('immutable prediction output must be new')
    if not receipts.get('supplement_prereg') or not receipts.get('main_parameter_seal'):
        raise ValueError('supplement registration and main frozen coefficient receipt required')
    directory.mkdir(parents=True)
    files={}
    for fold,doc in sorted(documents.items()):
        payload=canonical_bytes(doc)+b'\n';name=f'{fold}.joint_predictions.json'
        (directory/name).write_bytes(payload);files[name]=hashlib.sha256(payload).hexdigest()
    seal=dict(schema='joint211-all-four-gtfree-seal-v1',files=files,receipts=receipts)
    seal_bytes=canonical_bytes(seal)+b'\n'
    (directory/'ALL_FOUR_SEAL.json').write_bytes(seal_bytes)
    (directory/'ALL_FOUR_SEAL.sha256').write_text(hashlib.sha256(seal_bytes).hexdigest()+'\n')
    return seal


def verify_target_gate(directory,expected_receipts,*,cpu_grant):
    """Read numeric predictions only; never invoke this as an implicit GO."""
    if cpu_grant!='root-unique-joint211-field-cpu':
        raise PermissionError('root unique field CPU grant required')
    directory=Path(directory);payload=(directory/'ALL_FOUR_SEAL.json').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=(directory/'ALL_FOUR_SEAL.sha256').read_text().strip():
        raise ValueError('joint all-four top seal mismatch')
    seal=json.loads(payload)
    if seal['receipts']!=expected_receipts:raise ValueError('pinned registration/parameter receipt mismatch')
    if set(seal['files'])!={f'{f}.joint_predictions.json' for f in ('35','33','W','G')}:
        raise ValueError('all four prediction files must precede target GT')
    for name,sha in seal['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('prediction seal hash mismatch')
    return seal


def assert_reference(actual,reference,path=''):
    """Integer windows/graphs exact; native finite F1 floats tolerance 1e-12."""
    if isinstance(reference,dict):
        for k,v in reference.items():assert_reference(actual[k],v,path+'/'+k)
    elif isinstance(reference,list):
        if len(actual)!=len(reference):raise AssertionError(path+' length')
        for i,(a,b) in enumerate(zip(actual,reference)):assert_reference(a,b,path+'/'+str(i))
    elif isinstance(reference,float):
        if not math.isfinite(actual) or abs(actual-reference)>1e-12:raise AssertionError(path)
    elif actual!=reference:raise AssertionError(path)


def synthetic_selftest():
    """Small abstract state fixture; no native scorer or field reads."""
    actions={'c0':dict(family='C',canonical_ids=[1,2],mean_shared_iou=.9),
             'c1':dict(family='C',canonical_ids=[3,4],mean_shared_iou=.8),
             'i0':dict(family='I',canonical_ids=[2,5],min_local_norm=.1),
             'i1':dict(family='I',canonical_ids=[6,7],min_local_norm=.2)}
    import itertools
    states=[]
    for n in range(5):
        for ids in itertools.combinations(actions,n):
            s=dict(state_id='|'.join(ids) or 'K',selected_action_ids=list(ids))
            try:state_key(s,actions)
            except ValueError:continue
            states.append(s)
    # X dose (1,1): c0+i0 is forbidden even though it has best naive ranks.
    x=next(s for s in states if s['selected_action_ids']==['c1','i1'])
    z,proof=z_choice(states,actions,x)
    assert z['selected_action_ids']==['c0','i1'] and proof['rank_sum']==[1,1]
    u=lambda s:{p:float(len(s['selected_action_ids'])) for p in ('Q','L','X','V','C')}
    choices=choose_joint(states,actions,u)
    assert choices['choices']['K']=='K'
    assert state_key(choose_from_utilities(states,actions,lambda s:0),actions)==()
    assert 'C' in choices['frozen_X_family_only'] and 'I' in choices['frozen_X_family_only']
    return dict(status='PASS',legal_states=len(states),joint_Z=proof)


def native_synthetic_selftest(repo):
    """Deferred small actual-MOT/official-peer adapter check; no field files."""
    import sys
    import numpy as np
    sys.path.insert(0,str(repo));sys.path.insert(0,str(Path(__file__).parent))
    import joint_actions211 as joint
    import graphs211 as graphs
    from tools.study197.candidates import event_times,window_counts
    def row(f,p,x):return [f,p,x,-.5,1,1,.9]
    rows=np.array([row(1,1,-2),row(2,1,1),row(1,2,-2),row(2,2,1),
                   row(3,3,-2),row(1,4,-2),row(2,5,1)],float)
    canon=rows[:,1].astype(int);nf=40
    poly=np.array([[0,-10],[10,-10],[10,10],[0,10]],np.float32)
    count=lambda r:window_counts(event_times(r,poly),nf)
    baseline=count(rows);dummy=np.zeros_like(baseline);dummy[0,0]=-1
    co=dict(pair_id='1|2',canonical_ids=[1,2],shared_dates=[1,2],
            row_indices=[[0,2],[1,3]],mean_iou=1.)
    inter=[dict(pair_id='2|3',canonical_ids=[2,3],min_offset_norm=.1,delta_windows=dummy.tolist()),
           dict(pair_id='4|5',canonical_ids=[4,5],min_offset_norm=.2,delta_windows=dummy.tolist())]
    roster=joint.make_roster([co],['1|2'],{'1|2':dummy.tolist()},inter,['2|3','4|5'],
                             source_seals={'209':'a'*64,'210':'b'*64})
    deltas={a['action_id']:(count(joint.materialize(rows,canon,roster,[a['action_id']])[0])-baseline).tolist()
            for a in roster['actions']}
    for p in inter:p['delta_windows']=deltas['210:'+p['pair_id']]
    roster=joint.make_roster([co],['1|2'],{'1|2':deltas['209:1|2']},inter,['2|3','4|5'],
                             source_seals={'209':'a'*64,'210':'b'*64})
    peers=[]
    for shift in (100,200):
        r=rows.copy();r[:,1]+=shift;peers.append(r)
    unit=dict(unit='synthetic',rows=rows,canon=canon,polygon=poly,nframes=nf,peer_rows=peers)
    model={p:dict(a=.4,b=0. if p=='Q' else .3) for p in ('Q','L','X')}
    result=build_native_clip(unit,roster,model,joint=joint,graphs=graphs,count_windows=count)
    assert len(result['states'])==6
    assert len([s for s in result['states'] if s['deleted_raw_indices']])==2
    assert set(result['family_only'])=={'C','I'}
    # The main frozen utility is the same function, with its native nP key.
    import fit211
    K=next(s for s in result['states'] if not s['selected_action_ids'])
    ks=result['state_summaries'][K['state_id']]
    for sid,summary in result['state_summaries'].items():
        ours=frozen_utilities(model,summary,ks)
        theirs=fit211.utilities(model,dict(summary,nP=summary['nPred']),dict(ks,nP=ks['nPred']))
        assert ours=={k:v for k,v in theirs.items() if k!='K'}
    return dict(status='PASS',native_legal_states=6,actual_official_peer_graphs=12,
                all_physical_windows_additive=True,main_utilities_exact=True)
