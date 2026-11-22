"""Explicit independent §211 field adapter; imports perform no field IO."""
import sys
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction
from itertools import combinations
import hashlib
import json
import numpy as np
import audit211 as independent
sys.path.insert(0,str(Path(__file__).parents[1]/'study210'))
import audit210 as geometry
import field210 as oldfield
prior=oldfield.prior

PREREG='0e005ffeee692b104a4a6430843c0af721b6bfa1'
FROZEN='c21b74bbf3accd85a8a48863d16c1d241ce3045032b67ab8e6d92f390b88d6ae'
SOURCE206='066aaa245d78ee5d926cafeb709d200746e9ea9f736f820606270c35a2188d5d'
SOURCE208=oldfield.SOURCE208
SOURCE210='3810c1afabe45160642b6187c8fed89dac4e5e05dabb9f5be2cb696d666a7005'
DOMAINS=('35','33','W','G')
POLICIES=('K','Q','L','X','V','C','Z')

def array_sha(array):
    array=np.ascontiguousarray(array,dtype='<f8')
    return hashlib.sha256(json.dumps(list(array.shape)).encode()+b'\n'+array.tobytes()).hexdigest()

class OrderedGeometryCache:
    """Independent Hungarian, exact date+ordered ORIGINAL-source-rows cache.

    Prediction by_track order is supplied by the already prepared native event
    structure, avoiding re-sorting the same full matrix for three references.
    No reference labels or prediction IDs enter a cached IoU answer.
    """
    def __init__(self,reference,base,v,thresholds=('0.5','0.3')):
        self.reference=reference;self.base=base;self.v=v;self.alias=prior.canonical(reference)
        self.thresholds=thresholds
        self.frames=defaultdict(list);self.cache={};self.requests=0;self.order=None
        for j in np.lexsort((np.arange(len(reference)),reference[:,0],self.alias)):self.frames[int(reference[j,0])].append(j)
    def prepare(self,prediction):self.order=prediction['index']
    def owners(self,rows,raw_indices):
        from scipy.optimize import linear_sum_assignment
        assert self.order is not None
        frames=defaultdict(list)
        for j in self.order:frames[int(rows[j,0])].append(int(j))
        result={t:np.full(len(rows),-1,np.int64) for t in self.thresholds}
        for date,jj in frames.items():
            self.requests+=1;original=tuple(int(raw_indices[j]) for j in jj);key=date,original
            if key not in self.cache:
                gg=self.frames.get(date,[]);answer={t:np.full(len(jj),-1,np.int64) for t in result}
                if gg:
                    matrix=self.v.official_iou_independent(self.reference[gg,2:6],self.base[list(original),2:6]);a,b=linear_sum_assignment(-matrix)
                    for t in answer:
                        ok=matrix[a,b]>=float(t);answer[t][b[ok]]=self.alias[np.asarray(gg)[a[ok]]]
                self.cache[key]=answer
            for t in result:result[t][jj]=self.cache[key][t]
        return result

def load_frozen(args,seals):
    top=prior.read(args.root/'frozen/deployment.json');assert top['prereg']==PREREG and top['states']==22 and len(top['clips'])==6
    assert {(r['domain'],r['clip']) for r in top['clips']}=={(h,c) for h,c,s in prior.SPECS}
    assert top['source210']==SOURCE210 and top['source206']==SOURCE206 and top['source208']==SOURCE208
    assert prior.sha(args.source206/'manifest.json')==SOURCE206 and prior.sha(args.source208/'manifest.json')==SOURCE208
    assert prior.sha(args.catalog210/'manifest.json')==SOURCE210
    m206=prior.read(args.source206/'manifest.json');groups={r['unit']:r for r in prior.read(args.source206/'deployment.json')['groups']}
    m208=prior.read(args.source208/'manifest.json');records={};reports=[]
    for ref in top['clips']:
        key=ref['domain']+'/'+ref['clip'];folder=args.root/'frozen'/ref['path'];prior.seal(folder,ref['seal'])
        data=prior.read(folder/'states.json.gz');meta=data['metadata'];source=groups[meta['unit']]
        assert source==meta['peer206_group'];qpath=args.source206/source['path']/'queries.json';zpath=args.source206/source['path']/'K.npz'
        for p in (qpath,zpath):assert prior.sha(p)==m206[p.relative_to(args.source206).as_posix()]
        q=prior.read(qpath);sources=prior.read(folder/'sources.json.gz')
        with np.load(folder/'native.npz') as z:arrays={k:z[k] for k in z.files}
        base=arrays['receiver'];canon=arrays['receiver_canon'];assert base.dtype==np.float64 and canon.dtype==np.int64 and np.array_equal(prior.canonical(base),canon)
        source208=args.source208/key
        assert prior.sha(source208/'rows.npy')==m208[key+'/rows.npy']
        assert base.tobytes()==np.load(source208/'rows.npy',mmap_mode='r').tobytes()
        seed=3 if ref['domain'] in ('35','33') else 0
        with np.load(zpath) as z:
            assert base.tobytes()==z['k0'].tobytes() and sources['receiver']==q['members'][0]['source_keys'] and q['members'][0]['seed']==seed
            for slot in (1,2):
                assert arrays[f'peer{slot}'].tobytes()==z[f'k{slot}'].tobytes()
                assert sources['peers'][slot-1]==q['members'][slot]['source_keys'] and q['members'][slot]['seed']==seed+slot and q['members'][slot]['member_index']==slot
        catalog=prior.read(args.catalog210/key/'catalog.json.gz');pairs={p['pair_id']:p for p in catalog['pairs'] if p['active']}
        assert q['nf']==meta['nf']==catalog['metadata']['nf'] and q['polygon']==meta['polygon']==catalog['metadata']['polygon']
        assert {p['pair_id'] for p in data['active_pairs']}==set(pairs)
        assert set(pairs)==set(catalog['active_slate_ids']) and len({c for p in pairs.values() for c in p['canonical_ids']})==2*len(pairs)
        for p in data['active_pairs']:
            for f in ('canonical_ids','min_offset_norm','delta_windows'):assert p[f]==pairs[p['pair_id']][f]
        expected={frozenset(subset) for n in range(len(pairs)+1) for subset in combinations(pairs,n)}
        assert {frozenset(s['selected_pair_ids']) for s in data['states']}==expected and len(data['states'])==len(expected)
        assert len({s['state_id'] for s in data['states']})==len(expected)
        for state in data['states']:
            selected=state['selected_pair_ids'];assert state['edits']==len(selected)
            rows,order=geometry.merge(base,canon,[tuple(map(int,p.split('|'))) for p in selected],return_indices=True)
            with np.load(folder/state['materialization_file']) as z:
                assert z['output_raw_indices'].tolist()==order and np.array_equal(z['ids'],rows[:,1].astype(np.int64))
            assert array_sha(rows)==state['rows_sha256'] and sorted(order)==list(range(len(base)))
            assert np.array_equal(rows[:,[0,2,3,4,5,6]],base[order][:,[0,2,3,4,5,6]])
            if not selected:assert rows.tobytes()==base.tobytes()
        records[key]=dict(ref=ref,folder=folder,data=data,meta=meta,catalog_metadata=catalog['metadata'],base=base,canon=canon,peers=[arrays['peer1'],arrays['peer2']],pairs=pairs)
        reports.append(dict(clip=key,states=len(expected),rows=len(base),receiver_seed=seed,peer_seeds=[seed+1,seed+2],source206_exact=True,source208_exact=True,all_original_rows_once=True))
        print('AUDIT211_FROZEN',key,flush=True)
    assert sum(r['states'] for r in reports)==22
    prior.save(args.root/'verification/frozen.json',dict(passed=True,seals=seals,clips=reports,scope='All22 complete active subsets independently enumerated/materialized, original-row bijection/IDs/order and array hashes checked. Receiver208 and206slot0 exact bytes/sourcekeys; two peers206slot1/2 exact bytes/keys/seeds, no new member.'))
    return records

def validate_vertices(saved,structure):
    table={int(v['canonical_id']):v for v in saved}
    assert set(table)==set(structure['events'])
    for pid,(frame,direction) in structure['events'].items():
        vertex=table[pid];assert vertex['frame']==frame and vertex['direction']==direction and vertex['signed_frame']==frame*direction
        d,start,end,crosses=structure['tracks'][pid]
        assert vertex['start_row']==start and vertex['end_row']==end
        assert vertex['crossings']==[dict(canonical_row=int(j),frame=int(structure['rows'][j,0]),direction=int(sign)) for j,sign in crosses]
        j=next(j for j,sign in reversed(crosses) if sign==direction)
        prior.close(vertex['box'],structure['rows'][j,2:6],0)

def validate_peer(saved,calculation,isK):
    pred=calculation['prediction_structure'];ref=calculation['reference_structure'];graph=calculation['graphs']['0.5']
    validate_vertices(saved['prediction_vertices'],pred);validate_vertices(saved['reference_vertices'],ref)
    assert sorted(map(tuple,saved['edges']))==sorted(map(tuple,graph['edges']))
    assert saved['R']==graph['maxTP'] and saved['nPred']==len(graph['left']) and saved['nReference']==len(graph['right'])
    linked={p for p,g in graph['edges']};assert saved['J']==len(linked)
    assert np.array_equal(saved['pred_windows'],calculation['pred_windows']) and np.array_equal(saved['reference_windows'],calculation['reference_windows'])
    flags={f['canonical_id']:f for f in saved['flags']};assert set(flags)==set(graph['left'])
    marginal=independent.graph_diagnostics(graph['left'],graph['right'],graph['edges']) if isK else None
    for pid,f in flags.items():
        assert f['has_neighbor']==(pid in linked)
        if isK:assert f['drop_left_marginal']==marginal['marginal'][pid]
    return graph

def family_events(structure,raw_order,original_canon,pairs):
    parent_family={c:tuple(p['canonical_ids']) for p in pairs.values() for c in p['canonical_ids']};answer=[]
    for pid,(date,direction) in sorted(structure['events'].items()):
        _,start,end,crosses=structure['tracks'][pid];index=np.asarray(structure['index'])
        raw=[raw_order[int(j)] for j in index[start:end]];parents=sorted(set(int(original_canon[j]) for j in raw));family=set()
        for p in parents:family.update(parent_family.get(p,(p,)))
        chosen=[j for j,d in crosses if d==direction][-1]
        crossing=[dict(before_raw=int(raw_order[int(index[j-1])]),after_raw=int(raw_order[int(index[j])]),frame=int(structure['rows'][j,0]),direction=int(d)) for j,d in crosses]
        answer.append(dict(pid=int(pid),frame=int(date),direction=int(direction),signed_frame=int(date*direction),original_parent_ids=parents,family=sorted(family),selected_crossing_raw_index=int(raw_order[int(index[chosen])]),crossing_raw_indices=crossing))
    return answer

def native_stage(args,records,seals,helpers):
    from beetrack.evaluate import load_tracks
    replay,previous,native,v,_=helpers
    index=prior.read(args.root/'assessment/index.json');references={(r['domain'],r['clip']):r for r in index['clips']}
    oldGT=prior.read(args.labels208/'deployment.json')['GT_sha256'];reports=[];all_events=[]
    for key,r in records.items():
        meta=r['meta'];domain,name=r['ref']['domain'],r['ref']['clip'];ref=references[domain,name];entry=ref['GT_entry']
        gtpath=args.repo/entry['path'];assert prior.sha(gtpath)==entry['sha256']==oldGT[entry['path']]
        GT=load_tracks(gtpath);reference_rows=[GT,*r['peers']]
        structures=[native.track_structure(rows,meta['polygon']) for rows in reference_rows]
        caches=[OrderedGeometryCache(rows,r['base'],v,('0.5','0.3') if i==0 else ('0.5',)) for i,rows in enumerate(reference_rows)]
        actualpaths={s['state_id']:s for s in ref['states']};states={};Kfeatures=None
        for state in r['data']['states']:
            sid=state['state_id'];rows,order=geometry.merge(r['base'],r['canon'],[tuple(map(int,p.split('|'))) for p in state['selected_pair_ids']],return_indices=True)
            P=native.track_structure(rows,meta['polygon']);calculations=[]
            for i,(reference,cache) in enumerate(zip(structures,caches)):
                cache.prepare(P)
                calculations.append(independent.audit_native_state(rows,reference,meta['polygon'],meta['nf'],order,cache,native,geometry.clock,P,('0.5','0.3') if i==0 else ('0.5',)))
            current=calculations[0];peer_graphs=[]
            for saved,own in zip(state['peer_graphs'],calculations[1:]):peer_graphs.append(validate_peer(saved,own,state['is_K']))
            actualpath=args.root/'assessment'/actualpaths[sid]['path'];prior.seal(actualpath,actualpaths[sid]['seal'])
            actual=prior.read(actualpath/'score.json');metric=actual['metrics'];disk=np.loadtxt(actualpath/'mot.txt',delimiter=',',usecols=range(7),ndmin=2)
            assert disk.tobytes()==rows.tobytes() and actual['rows_sha256']==array_sha(rows)
            assert np.load(actualpath/'output_raw_indices.npy').tolist()==order
            assert np.array_equal(metric['pred_bins'],current['pred_windows']) and np.array_equal(metric['gt_bins'],current['reference_windows'])
            assert metric['nP']==len(P['events'])==state['nP'] and metric['nG']==len(structures[0]['events'])
            numerator=int(abs(current['pred_windows']-current['reference_windows']).sum());denominator=int(current['reference_windows'].sum())
            assert actual['numerator']==numerator and actual['denominator']==denominator
            for threshold,edge in (('0.5','edges05'),('0.3','edges03')):
                graph=current['graphs'][threshold];saved=actual['true_graphs'][threshold]
                assert sorted(map(tuple,metric['evidence'][edge]))==sorted(map(tuple,graph['edges']))==sorted(map(tuple,saved['edges']))
                for f in ('tp','pred','gt','tp_id'):assert metric['iou'][threshold][f]==current['iou'][threshold][f]
                assert saved['R']==graph['maxTP']
                assert {(a['pid'],a['signed_frame']) for a in saved['left_vertices']}=={(int(p),int(d*f)) for p,(f,d) in P['events'].items()}
                assert {(a['gt_event_id'],a['signed_frame']) for a in saved['right_vertices']}=={(int(p),int(d*f)) for p,(f,d) in structures[0]['events'].items()}
            events=family_events(P,order,r['canon'],r['pairs']);assert actual['family_events']==events
            J=[len({p for p,g in graph['edges']}) for graph in peer_graphs];M=[graph['maxTP'] for graph in peer_graphs]
            distances=[int(abs(c['pred_windows']-c['reference_windows']).sum()) for c in calculations[1:]]
            assert state['meanJ']==sum(J)/2 and state['meanM']==sum(M)/2 and state['meanCountL1']==sum(distances)/2
            assert state['peerCountL1']==distances and np.array_equal(state['pred_windows'],current['pred_windows'])
            if state['is_K']:
                features=independent.paired_predictors(list(P['events']),[dict(right=g['right'],edges=g['edges']) for g in peer_graphs])
                peer_diag=[independent.graph_diagnostics(g['left'],g['right'],g['edges']) for g in peer_graphs]
                marginal=independent.graph_diagnostics(current['graphs']['0.5']['left'],current['graphs']['0.5']['right'],current['graphs']['0.5']['edges'])
                table={f['canonical_id']:f for f in r['data']['K_event_features']};assert set(table)==set(features)
                Kfeatures=[]
                for pid,feature in sorted(features.items()):
                    f=table[pid];assert f['x_L']==feature['xL'] and f['x_X']==feature['xX']
                    date,direction=P['events'][pid];assert f['signed_frame']==date*direction and f['event_key']==f'{name}|{pid}|{date*direction}'
                    coverage=[dict(peer_index=i,nReference=len(g['right']),has_neighbor=peer_diag[i]['neighbor'][pid],drop_left_marginal=peer_diag[i]['marginal'][pid]) for i,g in enumerate(peer_graphs)]
                    Kfeatures.append(dict(domain=domain,clip=name,pid=int(pid),event_key=f'{domain}/{name}/{pid}',frozen_event_key=f['event_key'],signed_frame=int(date*direction),x_L=feature['xL'],x_X=feature['xX'],y=marginal['marginal'][pid],neighbor_peers=feature['neighbor_peers'],critical_peers=feature['critical_peers'],peercoverage=coverage))
                old=prior.read(args.reference210/'clips'/domain/name/'K/score.json')['metrics'];assert old==metric
            states[sid]=dict(state_id=sid,selected_pair_ids=state['selected_pair_ids'],edits=len(state['selected_pair_ids']),nP=state['nP'],meanJ=sum(J)/2,meanM=sum(M)/2,meanCountL1=sum(distances)/2,numerator=numerator,denominator=denominator,iou=current['iou'],graphs=current['graphs'],events=events,peer_graphs=peer_graphs)
            print('AUDIT211_NATIVE_STATE',key,sid,flush=True)
        assert Kfeatures is not None
        r.update(states=states,Kfeatures=Kfeatures);all_events.extend(Kfeatures)
        reports.append(dict(clip=key,states=len(states),peer_graphs=2*len(states),original_Kevents=len(Kfeatures),neighbor_peer_counts=dict(Counter(f['neighbor_peers'] for f in Kfeatures)),critical_peer_counts=dict(Counter(f['critical_peers'] for f in Kfeatures)),K_deletion_labels=dict(Counter(f['y'] for f in Kfeatures)),frame_requests=[c.requests for c in caches],unique_geometry_orders=[len(c.cache) for c in caches]))
        prior.save(args.root/'verification/native_clips'/domain/(name+'.json'),dict(passed=True,seals=seals,report=reports[-1]))
    labels=[]
    for domain in DOMAINS:
        shard=prior.read(args.root/'labels'/f'{domain}.json');assert shard['domain']==domain and shard['prereg']==PREREG and shard['frozen_manifest_sha256']==seals['frozen']
        own={e['event_key']:e['y'] for e in all_events if e['domain']==domain}
        assert {e['event_key']:e['y'] for e in shard['records']}==own
        labels.append(dict(domain=domain,events=len(own)))
    prior.save(args.root/'verification/graphs.json',dict(passed=True,seals=seals,clips=reports,source_label_shards=labels,scope='All22 fullscene actualGT Hungarian/event legal graphs bothIoU and two original-peer fullscene .5 graphs independently rebuilt, maxTP by independent DFS; everyK true/peer fixed-graph single-left deletion marginal and all source labels reconstructed. All raw states/MOT bijections/family/crossing provenance, complete30phase bins and originalK210 exact. Shared official low-level geometry and proven public197 track/event-edge semantics; no producer graphs/scorer imported. Peer references are predictions, not actualGT.'))
    return all_events

def model_stage(args,records,events,seals):
    predictions=prior.read(args.root/'predictions/predictions.json');reports=[];choices={};utilities={}
    for heldout in DOMAINS:
        model=prior.read(args.root/'predictions/models'/f'{heldout}.json');assert model['heldout']==heldout and set(model['source_domains'])==set(DOMAINS)-{heldout}
        expected=sorted([e for e in events if e['domain']!=heldout],key=lambda e:(e['domain'],e['clip'],e['pid']))
        training=model['training_records'];assert len(training)==len(expected)
        for actual,e in zip(training,expected):
            for f in ('event_key','domain','clip','pid','x_L','x_X','y','frozen_event_key','signed_frame','peercoverage'):assert actual[f]==e[f]
        weight=independent.source_weights(expected);prior.close([e['weight'] for e in training],weight)
        yy=np.asarray([e['y'] for e in expected]);q=float(np.dot(weight,yy)/weight.sum()) if len(expected) else .5
        if len(expected) and np.all(yy==yy[0]):q=float(yy[0])
        prior.close(q,model['Q']['a']);assert model['Q']['b']==0
        for head,feature in (('L','x_L'),('X','x_X')):
            own=independent.bounded_ls([e[feature] for e in expected],yy,weight);actual=model[head]
            prior.close([own['a'],own['b']],[actual['a'],actual['b']],2e-10)
            assert 0<=actual['a']<=1 and 0<=actual['b']<=1-actual['a']
            if len(expected):
                prior.close(own['objective'],actual['objective']);prior.close(own['gradient'],actual['gradient']);assert actual['KKT']['stationarity_norm']<1e-8
        normalized=dict(Q=model['Q']['a'],L=model['L'],X=model['X'])
        for key,r in records.items():
            if r['ref']['domain']!=heldout:continue
            saved=predictions['clips'][key]
            local={pid:p['min_offset_norm'] for pid,p in r['pairs'].items()}
            selected,values=independent.choose_states(list(r['states'].values()),normalized,local)
            selected['K']=next(sid for sid,s in r['states'].items() if not s['selected_pair_ids'])
            assert set(saved['selected'])==set(POLICIES)
            for policy,sid in selected.items():
                assert saved['selected'][policy]['state_id']==sid
                assert set(saved['selected'][policy]['selected_pair_ids'])==set(r['states'][sid]['selected_pair_ids'])
            savedstates={s['state_id']:s for s in saved['states']}
            for sid,ss in values.items():
                for policy,value in ss.items():
                    prior.close(value,savedstates[sid]['utilities'][policy]);assert round(value,10)==savedstates[sid]['rounded_utilities'][policy]
            choices[key]=selected;utilities[key]=values
        reports.append(dict(heldout=heldout,source_events=len(expected),source_domains=model['source_domains'],Q=model['Q']['a'],L=dict(a=model['L']['a'],b=model['L']['b']),X=dict(a=model['X']['a'],b=model['X']['b'])))
        print('AUDIT211_CALIBRATION',heldout,flush=True)
    prior.save(args.root/'verification/models.json',dict(passed=True,seals=seals,folds=reports,choices=choices,scope='All four other-three-hive source K-event cohorts, exact .5 graph-deletion y/peer predictors/coverage and hive→nonemptyclip→event weights reconstructed; Q mean and bounded-LS L/X independently solved by interior/three edges/vertices, coefficients/objective/gradient checked. All22-state relativeK utilities/round10/less-edit/numeric-pair ties and matched-X-dose Z choices independently reproduced.'))
    return choices,utilities

def diagnostic_stage(args,records,choices,utilities,seals):
    reports=[];pooled={h:{p:dict(N=0,D=0,iou={t:dict(tp=0,pred=0,gt=0) for t in ('0.5','0.3')}) for p in POLICIES} for h in DOMAINS}
    for key,r in records.items():
        states=r['states'];K=next(s for s in states.values() if not s['selected_pair_ids']);domain=r['ref']['domain'];name=r['ref']['clip']
        actual=prior.read(args.root/'assessment'/domain/name/'diagnostics.json');singletons=[];views=[]
        for sid,state in states.items():
            E=state['nP']+len(state['graphs']['0.5']['right'])-2*state['graphs']['0.5']['maxTP']
            views.append(dict(state_id=sid,selected=state['selected_pair_ids'],numerator=state['numerator'],event_error=E,F1=[2*state['graphs']['0.5']['maxTP'],max(1,state['nP']+len(state['graphs']['0.5']['right']))]))
            a=Counter((tuple(e['family']),e['signed_frame']) for e in K['events']);b=Counter((tuple(e['family']),e['signed_frame']) for e in state['events'])
            expand=lambda c:[dict(family=list(f),signed_frame=int(t),multiplicity=int(n)) for (f,t),n in sorted(c.items()) if n]
            saved=actual['signed_family_diffs'][sid]
            assert saved['unchanged']==expand(a&b) and saved['removed']==expand(a-b) and saved['added']==expand(b-a)
        for pid,pair in r['pairs'].items():
            state=next(s for s in states.values() if s['selected_pair_ids']==[pid]);parents=set(pair['canonical_ids']);saved=actual['singletons'][pid];checks={}
            old=[e for e in K['events'] if set(e['original_parent_ids'])<=parents and set(e['original_parent_ids'])&parents]
            new=[e for e in state['events'] if set(e['original_parent_ids'])<=parents and set(e['original_parent_ids'])&parents]
            for t in ('0.5','0.3'):
                before=K['graphs'][t];after=state['graphs'][t]
                old_simple=[dict(id=e['pid'],direction=e['direction']) for e in old];new_simple=[dict(id=e['pid'],direction=e['direction']) for e in new]
                strict=independent.strict_duplicate(before,after,old_simple,new_simple)
                assert strict==saved[t]['strict_event_redundancy_witness']
                km=K['iou'][t];sm=state['iou'][t]
                delta=dict(tp=sm['tp']-km['tp'],fp=(sm['pred']-sm['tp'])-(km['pred']-km['tp']),fn=(sm['gt']-sm['tp'])-(km['gt']-km['tp']))
                delta['E']=delta['fp']+delta['fn'];assert delta==saved[t]['delta']
                for ev in saved[t]['before_parent_events']:
                    assert ev['legal_GT_neighbors']==sorted(g for p,g in before['edges'] if p==ev['pid'])
                for ev in saved[t]['new_family_events']:
                    assert ev['legal_GT_neighbors']==sorted(g for p,g in after['edges'] if p==ev['pid'])
                checks[t]=dict(strict_official_event_redundancy=strict,delta=delta)
            assert saved['delta_window_numerator']==state['numerator']-K['numerator']
            singletons.append(dict(pair_id=pid,delta_window_numerator=state['numerator']-K['numerator'],thresholds=checks))
        front=independent.pareto(views);opt=min(views,key=lambda s:(s['numerator'],len(s['selected']),tuple(sorted(tuple(map(int,p.split('|'))) for p in s['selected']))))
        baseline=next(v for v in views if v['state_id']==K['state_id']);inversions=[v['state_id'] for v in views if v['numerator']<baseline['numerator'] and v['event_error']>baseline['event_error']]
        policy={}
        for p in POLICIES:
            sid=choices[key][p];state=states[sid];m=pooled[domain][p];m['N']+=state['numerator'];m['D']+=state['denominator']
            for t in m['iou']:
                for f in ('tp','pred','gt'):m['iou'][t][f]+=state['iou'][t][f]
            E=state['nP']+len(state['graphs']['0.5']['right'])-2*state['graphs']['0.5']['maxTP']
            rawutility=0. if p in ('K','Z') else utilities[key][sid][p]
            policy[p]=dict(state_id=sid,selected=state['selected_pair_ids'],N=state['numerator'],D=state['denominator'],E=E,delta_E=E-baseline['event_error'],utility=None if p=='Z' else rawutility,positive_utility_but_E_worse=p not in ('K','Z') and round(rawutility,10)>0 and E>baseline['event_error'])
        reports.append(dict(clip=key,states=views,Pareto_state_ids=[v['state_id'] for v in front],minimum_window_witness=opt,window_improves_E_worsens=inversions,singletons=singletons,policies=policy))
    for h,pp in pooled.items():
        for p,m in pp.items():
            m['W_fraction']=[100*m['N'],m['D']];m['F1_fraction']={t:[2*c['tp'],max(1,c['pred']+c['gt'])] for t,c in m['iou'].items()}
            m['E']={t:c['pred']+c['gt']-2*c['tp'] for t,c in m['iou'].items()}
    W=lambda h,p:Fraction(*pooled[h][p]['W_fraction']);F=lambda h,p:Fraction(*pooled[h][p]['F1_fraction']['0.5'])
    gates=dict(head35_gain_at_least1=W('35','K')-W('35','X')>=1,head35_strictly_beats_L_Z=all(W('35','X')<W('35',p) for p in ('L','Z')),head33_loss_at_most_halfpoint=W('33','X')-W('33','K')<=Fraction(1,2),headF1_no_harm_hundredth=all(F(h,'X')>=F(h,'K')-Fraction(1,100) for h in ('35','33')),WG_W_no_harm=all(W(h,'X')<=W(h,'K') for h in ('W','G')))
    mechanism={}
    for h in ('35','33'):
        row=next(r for r in reports if r['clip'].startswith(h+'/'));x,l=row['policies']['X'],row['policies']['L']
        mechanism[h]=dict(different_configuration=set(x['selected'])!=set(l['selected']),E_X_strictly_better=x['E']<l['E'])
    prior.save(args.root/'verification/diagnostics.json',dict(passed=True,seals=seals,clips=reports,pooled=pooled,investment=dict(checks=gates,passed=all(gates.values())),capacity_attribution=dict(heads=mechanism,passed=all(all(v.values()) for v in mechanism.values())),scope='All22 true window/E/F1 tradeoffs and perclip Pareto, exact minimumW tie-witness actualF1, every singleton trueTP/FP/FN/E/window change and conservative official-event duplicate witness checked. Every state family/signed-date multiset, sevenpolicy choices and per-hive raw pooled metrics/gates rebuilt. Count extrema alone never supply F1; peer predictions/weak labels do not establish same physical bee.'))

def run(args):
    if not args.field_go or not args.fit_finished:raise ValueError('root soleCPU GO and complete fourfold fit release required')
    sys.path.insert(0,str(args.repo.resolve()))
    pins=prior.read(args.seal_pins);seals={stage:prior.seal(args.root/stage,digest) for stage,digest in pins.items()}
    assert set(seals)=={'frozen','assessment','labels','predictions'} and seals['frozen']==FROZEN
    assert prior.read(args.root/'predictions/receipt.json')['frozen_manifest_sha256']==seals['frozen']
    replay=prior.module('audit211_public203',args.repo/'tools/study203/verify203.py');helpers=(replay,*replay.helpers(args.repo))
    records=load_frozen(args,seals);events=native_stage(args,records,seals,helpers)
    choices,utilities=model_stage(args,records,events,seals);diagnostic_stage(args,records,choices,utilities,seals)
    files=('frozen.json','graphs.json','models.json','diagnostics.json');reports={f:prior.sha(args.root/'verification'/f) for f in files}
    sources={Path(__file__).name:prior.sha(__file__),Path(independent.__file__).name:prior.sha(independent.__file__)}
    snapshot=args.root/'verification/source_snapshot';snapshot.mkdir(exist_ok=True)
    for f in sources:(snapshot/f).write_bytes(Path(__file__).with_name(f).read_bytes())
    prior.save(args.root/'verification/independent.json',dict(passed=True,prereg=PREREG,seals=seals,reports=reports,source_sha256=sources,reused_independent_sources={'audit210.py':prior.sha(geometry.__file__),'field210.py':prior.sha(oldfield.__file__),'audit209.py':prior.sha(prior.__file__)},scope='Original22 only; full independent raw/materialization, fresh geometry+legal graph and independent maxmatch/deletion/calibration/choices/diagnostics as stage receipts specify. Native event semantics and low-level IoU/Hungarian are shared proven kernels; no producer helper used. No joint-family field read.'))
    print('AUDIT211_PASS',prior.sha(args.root/'verification/independent.json'),flush=True)
