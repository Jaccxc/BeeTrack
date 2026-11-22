"""Small post-native crosschecks; only completed numeric JSON, no GT/MOT IO."""
import argparse,json,hashlib
from pathlib import Path
import verify_joint_stage211 as core

def run(args):
    own=core.read(args.verification/'independent.json');core.sealed(args.verification,args.verification_seal)
    job=core.read(args.job);core.sealed(job['score_out'],own['assessment_seal'])
    scored=core.read(Path(job['score_out'])/'assessment.json');summary=core.read(args.summary/'summary.json')
    core.sealed(args.summary,args.summary_seal)
    for view in ('joint','C','I'):
        table=own['domains'] if view=='joint' else {h:own['family_domains'][h][view] for h in core.DOMAINS}
        for h,policies in table.items():
            for p,r in policies.items():
                s=summary['views'][view][h][p]
                assert (s['numerator'],s['denominator'])==(r['numerator'],r['denominator'])
                for field,k in (('gt','gt'),('pred','pred'),('tp05','tp')):assert s[field]==r['iou']['0.5'][k]
                assert s['tp03']==r['iou']['0.3']['tp']
    contrasts=[];states_checked=0
    for c in scored['clips']:
        h,name=c['domain'],c['clip'];r=core.read(args.verification/(h+'_'+name+'.json'));K=next(s for s in r['states'] if not s['selected_action_ids'])
        split={'pure_I_including_K':[],'contains_C':[]}
        for state in r['states']:
            sid=state['state_id'];actual=c['states'][sid];metric=actual['metrics'];events={(int(k),int(f*d)) for k,(f,d) in state['events'].items()}
            assert events=={tuple(v[:2]) for v in metric['evidence']['predictions']}
            for t in ('0.5','0.3'):
                g=actual['true_graphs'][t]
                assert events=={(v['pid'],v['signed_frame']) for v in g['left_vertices']}
                assert {tuple(v[:2]) for v in metric['evidence']['gt']}=={(v['gt_event_id'],v['signed_frame']) for v in g['right_vertices']}
                graph=state['graphs'][t]
                for f in g['flags']:assert f['has_neighbor']==any(p==f['canonical_id'] for p,q in graph['edges'])
                matching=list(map(tuple,g['matching']));assert set(matching)<=set(map(tuple,graph['edges']))
                assert len(matching)==len({p for p,q in matching})==len({q for p,q in matching})==graph['maxTP']
            value=(state['meanJ']-K['meanJ'])-(state['meanM']-K['meanM'])
            key='contains_C' if any(r['actions'][a]['family']=='C' for a in state['selected_action_ids']) else 'pure_I_including_K'
            split[key].append(value);states_checked+=1
        stats={key:dict(states=len(v),nonzero=sum(x!=0 for x in v),min=min(v,default=None),max=max(v,default=None)) for key,v in split.items()}
        strict={t:sum(x['strict_duplicate'][t]['qualifies'] for x in r['strict_duplicate_transitions']) for t in ('0.5','0.3')}
        contrasts.append(dict(domain=h,clip=name,contrast=stats,strict_duplicate_transitions=strict))
    out=args.out
    if out.exists():raise FileExistsError('new output only')
    out.mkdir(parents=True);report=dict(schema='joint211-independent-numeric-crosschecks-v1',passed=True,states_checked=states_checked,all_three_views_28_policy_domain_tables_match=True,contrasts=contrasts,verification_seal=args.verification_seal,summary_seal=args.summary_seal,code_sha256=core.sha(__file__))
    (out/'numeric.json').write_bytes(core.wire(report)+b'\n');(out/'manifest.json').write_bytes(core.wire({'numeric.json':core.sha(out/'numeric.json')})+b'\n');(out/'SEALED.sha256').write_text(core.sha(out/'manifest.json')+'\n')
    print(core.wire(report).decode());print('NUMERIC_SEAL',core.sha(out/'manifest.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);p.add_argument('--verification-seal',required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--summary-seal',required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
