"""Independent small diagnostic completion and durable212 receipt."""
import argparse,sys,shutil
from pathlib import Path
from collections import Counter,defaultdict
import field212_models as m
import numpy as np
p=m.prior

def run(a):
    assert a.field_go
    sys.path.insert(0,str(a.repo.resolve()))
    E=p.read(a.root/'evaluation/summary.json');P=p.read(a.root/'predictions/predictions.json')
    top,records=m.source.load_records(a);bank=p.read(a.root/'frozen'/top['feature_bank_file'])['features']
    own=p.read(a.root/'verification/evaluation.json');assert own['passed']
    for domain,means in E['Brier_member_means']['domain_member_means'].items():
        for h,value in means.items():
            cells=[x for x in E['Brier_member_means']['pooled_original_members'] if x['domain']==domain]
            assert len(cells)==3
            p.close(sum(x['Brier'][h] for x in cells)/3,value,1e-12)
    changes={};sourcechecks=0
    for key,r in records.items():
        c=P['clips'][key];states={s['state_id']:s for s in c['states']};K=states[c['K_state_id']]
        metrics={sid:p.read(a.root/'evaluation/states'/key/(sid+'.json'))['metrics'] for sid in states}
        declared=next(x for x in E['clips'] if x['domain']==c['domain'] and x['clip']==c['clip'])['event_changes']
        for arm in m.POLICIES:
            s=states[c['selected'][arm]['state_id']];families={int(pid):tuple(sorted(map(int,action['canonical_ids']))) for action in s['actions'] for pid in action['canonical_ids']}
            def annotate(state):
                rows=[];ev=metrics[state['state_id']]['evidence']
                for e in state['events']:
                    fam=tuple(sorted({q for pid in e['original_parent_ids'] for q in families.get(int(pid),(int(pid),))}))
                    neighbors={t:sorted(int(g) for pid,g in ev[k] if int(pid)==e['canonical_id']) for t,k in [('0.5','edges05'),('0.3','edges03')]}
                    rows.append((e['event_ref'],fam,int(e['signed_frame']),neighbors))
                return rows
            b,after=annotate(K),annotate(s);bc=Counter((f,t) for _,f,t,_ in b);ac=Counter((f,t) for _,f,t,_ in after)
            for label,rows,remaining in [('before',b,bc-ac),('after',after,ac-bc)]:
                actual={e['event_ref']:e for e in declared[arm][label]};assert len(rows)==len(actual)
                for ref,f,t,neighbors in sorted(rows):
                    d=actual[ref];changed=remaining[f,t]>0
                    if changed:remaining[f,t]-=1
                    assert d['comparison_family']==list(f) and d['signed_frame']==t and d['changed']==changed and d['legal_GT_neighbors']==neighbors
            assert declared[arm]['new_events_without_legal_edge05']==sum(x['changed'] and not x['legal_GT_neighbors']['0.5'] for x in declared[arm]['after'])
        changes[key]=True
    for heldout in ('35','33','W','G'):
        model=p.read(a.root/'predictions/models'/f'{heldout}.json')
        for key,c in P['clips'].items():
            if c['domain']==heldout:continue
            states=c['states'];K=next(s for s in states if s['state_id']==c['K_state_id'])
            H=lambda s:s['nPred']-2*sum(m.probability(model,bank[e['feature_key']],'T') for e in s['events'])
            base=H(K);kg=p.read(a.root/'evaluation/states'/key/(K['state_id']+'.json'));ko=m.graph_outcome(kg['metrics'],records[key]['meta']['nf'])
            for s in states:
                if s['state_id']==K['state_id']:continue
                row=next(x for x in E['source_guard_diagnostics']['changes'] if (x['heldout'],x['source_domain'],x['clip'],x['state_id'])==(heldout,c['domain'],c['clip'],s['state_id']))
                actual=m.graph_outcome(p.read(a.root/'evaluation/states'/key/(s['state_id']+'.json'))['metrics'],records[key]['meta']['nf'])
                dh=H(s)-base;de=actual['iou']['0.5']['E']-ko['iou']['0.5']['E'];accepted=dh<=1e-9
                p.close(dh,row['T_delta_Ehat'],1e-12);assert row['accepted']==accepted and row['actual_delta_E05']==de
                category=('accepted_' if accepted else 'rejected_')+('actual_harm' if de>0 else 'actual_improvement' if de<0 else 'equal_actual_E')
                assert category==row['category'];sourcechecks+=1
    assert sourcechecks==len(E['source_guard_diagnostics']['changes'])
    assert dict(Counter(x['category'] for x in E['source_guard_diagnostics']['changes']))==E['source_guard_diagnostics']['counts']
    p.save(a.root/'verification/supplement.json',dict(passed=True,all_state_local_family_signed_multiplicity_and_neighbors=changes,source_guard_posthoc_checks=sourcechecks,domain_member_means=True,scope='Diagnostic replay only; no source diagnostic entered fitting or frozen selection.'))
    folder=a.root/'verification';snap=folder/'source_snapshot_final';snap.mkdir(exist_ok=True);shutil.copyfile(__file__,snap/Path(__file__).name)
    sources={str(f):p.sha(f) for f in Path(__file__).parent.glob('*.py') if f.name in ('audit212.py','field212_features.py','field212_truth.py','field212_models.py','finalize212.py')}
    reports={n:p.sha(folder/n) for n in ('features.json','labels.json','models.json','selections.json','evaluation.json','supplement.json')}
    p.save(folder/'independent.json',dict(schema='phase212-independent-v1',passed=True,prereg=m.source.PREREG,input_seals=m.source.PINS,reports=reports,sources=sources,scope={'original_physical_carriers_and_date_side_features':'all86 states plus18 original K;1058 unique sequences, all16 salts','fresh_GT_matching':'all18 original K and5 original211 singleton states at both IoUs; official lowlevel geometry kernel shared, independent fullscene owners/maxmatching','old86_outcomes':'old legal graphs/maxmatching and30phase replay, zero86 fresh MOT scorers','models':'12 independent source-only receiverK G/R/T refits; S shares T weights/scaler; all state and18K inference','policies':'all guards/round10/C ties/six policies/raw integer pool and gates','limitation':'Annotation legal-edge target is not biological identity, matched TP, or deletion criticality. Reserve members share two head clips/GT and are correlated.'},primary_gate=own['primary_gate'],phase_attribution=own['phase_attribution']))
    print('AUDIT212_FINAL_PASS',p.sha(folder/'independent.json'),flush=True)

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--repo',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--field-go',action='store_true');run(q.parse_args())
