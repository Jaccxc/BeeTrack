"""Private phase1 peer-gate restricted-count capacity and actual-witness F1.

Pure core consumes normalized numeric units; no GT/image/trajectory loaders.
Count lowerbound is optimistic for anF1 guard; only this actual count-witness's
F1 is reported. A good count bound never proves anF1-constrained solution exists.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from tools.study197.oracle import solve_clip,phase_counts


def _coverage(candidates):
    counts=dict(all_candidates=len(candidates),frame_only_gate_candidates=sum(c.get('frame_only_gate',False) for c in candidates))
    for sign,label in ((1,'add'),(-1,'drop')):
        pool=[c for c in candidates if c['delta']==sign]
        kept=[c for c in pool if c['qualifies']]
        counts['all_'+label]=len(pool);counts['qualified_'+label]=len(kept)
        counts['fraction_'+label]=len(kept)/len(pool) if pool else None
    # Optional prediction-only metadata. These counts never affectqualifies.
    # Coverage is framelayer withoutID and is not evidence ofsame-bee visibility.
    counts['frame_only_good_frames']=sum(int(c.get('frame_only_observable_count',c.get('frame_only_good_frames',0))) for c in candidates)
    counts['short_frames']=sum(int(c.get('total_short_frames',c.get('short_frames',0))) for c in candidates)
    counts['frame_only_fraction']=counts['frame_only_good_frames']/counts['short_frames'] if counts['short_frames'] else None
    return counts


def _f1(members):
    tp=sum(m['tp'] for m in members);pred=sum(m['pred'] for m in members);gt=sum(m['gt'] for m in members)
    return dict(tp=tp,pred=pred,gt=gt,F1=2*tp/(pred+gt) if pred+gt else None)


def _member_mean_f1(rows):
    """Registered198:pool clips within member,then mean3member F1."""
    grouped={}
    for row in rows:grouped.setdefault(row['member_index'],[]).append(row)
    members={str(i):_f1(values) for i,values in grouped.items()}
    if len(members)!=3:raise ValueError('F1 requires exactly3 member identities across groupclips')
    values=[m['F1'] for m in members.values()]
    return dict(members=members,mean_member_F1=float(np.mean(values)) if all(v is not None for v in values) else None,
                pooled_diagnostic=_f1(rows))


def run_units(units,score_witness=None,time_limit=60):
    """unit:{unit,hive,group,video,nf,pred_sum,gt_bins,candidates}.

    Eachcandidate inherits197 fields plus member_index andqualifies bool.
    Optional score_witness(unit,selected_ids)->{pred_sum,members:[{member_index,
    tp,pred,gt}]} must use labels198's legal K/A graphs for the SAMEphysicalmask.
    No callback means phase1 count-only resource screening,F1 explicitlyunknown.
    """
    results={};groups={}
    for unit in units:
        key=unit['unit']
        if key in results:raise ValueError('duplicateunit')
        candidates=unit['candidates']
        if any(type(c['qualifies']) is not bool for c in candidates):
            raise ValueError('qualifies must be frozen producerbool,not implicit truthiness')
        filtered=[c for c in candidates if c['qualifies']]
        optimum=solve_clip(unit['pred_sum'],unit['gt_bins'],filtered,unit['nf'],time_limit=time_limit)
        assert set(optimum['selected_ids'])<=set(c['candidate_id'] for c in filtered)
        witness=None
        if score_witness is not None:
            witness=score_witness(unit,optimum['selected_ids'])
            bins=np.asarray(witness['pred_sum'],np.int64)
            assert np.array_equal(bins,np.asarray(optimum['witness_pred_sum'],np.int64))
            assert int(np.abs(bins-3*np.asarray(unit['gt_bins'],np.int64)).sum())==optimum['upper_bound']
            witness['pooled_diagnostic']=_f1(witness['members'])
        member_indices=unit.get('member_indices',list(range(3)))
        members={str(m):_coverage([c for c in candidates if c['member_index']==m]) for m in member_indices}
        member_masks={str(m):[c['candidate_id'] for c in filtered if c['member_index']==m and c['candidate_id'] in set(optimum['selected_ids'])]
                      for m in member_indices}
        report=dict(unit=key,hive=str(unit['hive']),group=unit['group'],video=unit['video'],
                    coverage=_coverage(candidates),members=members,member_masks=member_masks,oracle=optimum,witness=witness,
                    baseline_pred_sum=unit['pred_sum'],gt_bins=unit['gt_bins'])
        results[key]=report
        group=groups.setdefault((str(unit['hive']),unit['group']),dict(videos=[],baseline=0,lower=0,upper=0,denominator=0,
                              coverage={f:0 for f in ('all_candidates','frame_only_gate_candidates','all_add','all_drop','qualified_add','qualified_drop','frame_only_good_frames','short_frames')},witness_members=[]))
        group['videos'].append(key)
        for field,source in (('baseline','baseline_numerator'),('lower','lower_bound'),('upper','upper_bound'),('denominator','denominator')):
            group[field]+=optimum[source]
        for field in group['coverage']:group['coverage'][field]+=report['coverage'][field]
        if witness is not None:group['witness_members'].extend(witness['members'])
    hives={}
    for key,group in groups.items():
        d=group['denominator'];group['optimal']=group['lower']==group['upper'];group['gap']=group['upper']-group['lower']
        group['percent']={k:100*group[k]/d if d else None for k in ('baseline','lower','upper')}
        group['gain_ceiling']=100*(group['baseline']-group['lower'])/d if d else None
        group['witness_gain']=100*(group['baseline']-group['upper'])/d if d else None
        group['witness_F1']=_member_mean_f1(group['witness_members']) if score_witness else None
    for hive in sorted({h for h,g in groups}):
        rows=[g for(h,_),g in groups.items() if h==hive];defined=all(g['denominator']>0 for g in rows)
        ceiling_exact=sum(Fraction(100*(g['baseline']-g['lower']),g['denominator']) for g in rows)/len(rows) if defined else None
        attained_exact=sum(Fraction(100*(g['baseline']-g['upper']),g['denominator']) for g in rows)/len(rows) if defined else None
        ceiling=float(ceiling_exact) if defined else None;attained=float(attained_exact) if defined else None
        line={'35':1.4,'33':1.3}.get(hive)
        line_exact=Fraction(str(line)) if line is not None else None
        status=('witness_exceeds_line' if attained_exact>line_exact else 'excluded_by_lower_bound' if ceiling_exact<=line_exact else 'unknown_gap') if defined and line is not None else 'not_gated'
        hives[hive]=dict(groups=len(rows),defined=defined,all_count_optimal=all(g['optimal'] for g in rows),
            mean_gain_ceiling=ceiling,mean_witness_gain=attained,reference_line=line,
            count_can_exceed_line=ceiling_exact>line_exact if defined and line is not None else None,
            count_witness_exceeds_line=attained_exact>line_exact if defined and line is not None else None,
            count_space_status=status,
            gain_ceiling_fraction=[ceiling_exact.numerator,ceiling_exact.denominator] if defined else None,
            witness_gain_fraction=[attained_exact.numerator,attained_exact.denominator] if defined else None,
            mean_witness_F1=float(np.mean([g['witness_F1']['mean_member_F1'] for g in rows])) if score_witness and all(g['witness_F1']['mean_member_F1'] is not None for g in rows) else None,
            coverage={f:sum(g['coverage'][f] for g in rows) for f in rows[0]['coverage']})
    statuses=[hives[h]['count_space_status'] for h in ('35','33') if h in hives]
    nominal=('witnesses_exceed_both_lines' if len(statuses)==2 and all(s=='witness_exceeds_line' for s in statuses)
             else 'excluded' if any(s=='excluded_by_lower_bound' for s in statuses) else 'unknown_or_incomplete')
    flat=[dict(domain=h,group=gid,K_numerator=v['baseline'],O_upper=v['upper'],O_lower=v['lower'],denominator=v['denominator']) for(h,gid),v in groups.items()]
    return dict(units=results,groups={f'{h}/{g}':v for(h,g),v in groups.items()},group_records=flat,hives=hives,
                count_gate=nominal,resource_gate='Await all60 physical-mask validations;this file does not authorize stage2',
                F1_scope='Actual count-oracle witness only;noF1-constrained optimality claim',
                frame_diagnostic_scope='Producer frame_only_gate reference plus candidate-short-frame occurrences;withoutID,never expands gate')


def normalize_peer(peer,deployment197,summary197):
    """Join sealed GT-freepeerrecords with ONLY197's completed numericbaseline.

    The caller pins allthree hashes. Peerframe-only coverage never sets the
    frozenqualifies bit. Retain alloriginalcandidates forcoverage,filteronly
    when solving. Original vectors and member ownership must remain exact.
    """
    source={u['unit']:u for u in deployment197['groups']};units=[]
    if {u['unit'] for u in peer['groups']}!=set(source):
        raise ValueError('peer and197 units differ')
    for group in peer['groups']:
        original=source[group['unit']];hive=str(original['domain']);video=original['video'];gid=original['group']
        score=next(c for g in summary197['domains'][hive]['methods']['K']['groups'] if g['group']==gid for c in g['clips'] if c['video']==video)
        candidates=[];member_indices=[]
        for member in group['members']:
            index=member['member_index'];member_indices.append(index)
            baseline=next(m for m in original['members'] if m['member_index']==index)
            old={c['candidate_id']:c for c in baseline['bundle']['candidates']}
            records=member['records']
            if {r['candidate_id'] for r in records}!=set(old):raise ValueError('peerrecords do not cover exact197 candidateinventory')
            for record in records:
                prior=old[record['candidate_id']]
                if type(record['qualifies']) is not bool:raise ValueError('strict frozenboolrequired')
                if 'delta_windows' in record and not np.array_equal(record['delta_windows'],prior['delta_windows']):
                    raise ValueError('peer altered197 eventcolumns')
                candidates.append(dict(prior,member_index=index,qualifies=record['qualifies'],
                    frame_only_gate=record['frame_only_gate'],total_short_frames=record['total_short_frames'],frame_only_observable_count=record['frame_only_observable_count']))
        if len(member_indices)!=3 or len(set(member_indices))!=3:raise ValueError('exactly3 distinct originalmembers required')
        units.append(dict(unit=group['unit'],hive=hive,group=gid,video=video,nf=original['nf'],
                          pred_sum=score['sum_pred_bins'],gt_bins=score['gt_bins'],candidates=candidates,
                          member_indices=member_indices))
    return dict(prereg=peer['prereg'],units=units,graph_data=peer.get('graph_data'))


def finalize_capacity(count,physical_audit,audit_receipt):
    """After all60 native/graph checks,seal the phase1resource decision.

    Physical failedF1 is reported only for this actual witness;it cannot prove
    absence of anotherF1-admissible count solution. No model loading here.
    """
    audit=physical_audit['members'];expected={(unit,int(member)) for unit,row in count['units'].items() for member in row['member_masks']}
    masks={unit:row['member_masks'] for unit,row in count['units'].items()}
    masks_digest=hashlib.sha256(json.dumps(masks,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if physical_audit['requested_masks_sha256']!=masks_digest:raise ValueError('physical audit not tied toexactselected masks')
    observed={(row['unit'],int(row['member_index'])) for row in audit}
    if len(audit)!=60 or len(expected)!=60 or observed!=expected or physical_audit['complete_members']!=60:
        raise ValueError('exactly all60 originalmembers must have a native physical audit')
    if any(not row['native_verified'] or not row['graph_verified'] for row in audit):
        raise ValueError('native AND K/A graph checks required for all60')
    groups={}
    for unit,row in count['units'].items():
        records=[m for m in audit if m['unit']==unit]
        pred=sum(np.asarray(m['pred_bins'],np.int64) for m in records)
        gt=np.asarray(row['gt_bins'],np.int64)
        assert all(np.array_equal(np.asarray(m['gt_bins']),gt) for m in records)
        assert np.array_equal(pred,np.asarray(row['oracle']['witness_pred_sum']))
        assert int(np.abs(pred-3*gt).sum())==row['oracle']['upper_bound']
        key=(row['hive'],row['group']);group=groups.setdefault(key,[])
        for m in records:
            metric=m['iou']['0.5']
            group.append(dict(member_index=m['member_index'],tp=metric['tp'],pred=m['nP'],gt=m['nG']))
    for(hive,gid),members in groups.items():
        count['groups'][f'{hive}/{gid}']['native_witness_F1']=_member_mean_f1(members)
    for hive in count['hives']:
        values=[_member_mean_f1(m)['mean_member_F1'] for(h,_),m in groups.items() if h==hive]
        count['hives'][hive]['mean_native_witness_F1']=float(np.mean(values)) if all(v is not None for v in values) else None
    statuses=[count['hives'][h]['count_space_status'] for h in ('35','33')]
    gate='pass' if all(s=='witness_exceeds_line' for s in statuses) else 'fail_count_space' if any(s=='excluded_by_lower_bound' for s in statuses) else 'unknown_gap_stop'
    count['resource_gate']=gate;count['physical_audit_receipt']=audit_receipt;count['all60_native_graph_verified']=True
    count['physical_verified_members']=60
    count['stage2_authorized_by_count_gate']=gate=='pass'
    return count


def selfcheck():
    nf=26;gt=phase_counts([(1,1)],nf);pred=3*gt
    candidates=[dict(candidate_id='qualified_add',member_index=0,frame=26,direction=1,delta=1,qualifies=True),
                dict(candidate_id='unqualified_drop',member_index=1,frame=1,direction=1,delta=-1,qualifies=False)]
    unit=dict(unit='synthetic',hive='35',group=1,video='synthetic',nf=nf,pred_sum=pred.tolist(),gt_bins=gt.tolist(),candidates=candidates)
    def scorer(unit,selected):
        assert selected==[]
        return dict(pred_sum=unit['pred_sum'],members=[dict(member_index=i,tp=1,pred=1,gt=1) for i in range(3)])
    result=run_units([unit],scorer)
    assert result['units']['synthetic']['oracle']['selected_ids']==[]
    assert result['hives']['35']['mean_witness_F1']==1 and result['hives']['35']['mean_gain_ceiling']==0
    assert result['hives']['35']['coverage']['qualified_drop']==0
    noF1=run_units([unit]);assert noF1['hives']['35']['mean_witness_F1'] is None
    metric=_member_mean_f1([dict(member_index=0,tp=1,pred=1,gt=1),dict(member_index=0,tp=0,pred=1,gt=0),
        dict(member_index=1,tp=1,pred=1,gt=1),dict(member_index=1,tp=1,pred=1,gt=1),
        dict(member_index=2,tp=0,pred=100,gt=1),dict(member_index=2,tp=0,pred=100,gt=1)])
    assert abs(metric['mean_member_F1']-5/9)<1e-12
    assert metric['pooled_diagnostic']['F1']!=metric['mean_member_F1']
    print('Synthetic restricted gate,countcertificate,actualmask scorer andnoF1 phase1 passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--sha256');p.add_argument('--prereg')
    p.add_argument('--graph-adapter',type=Path);p.add_argument('--phase1',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--source-deployment197',type=Path);p.add_argument('--source-summary197',type=Path)
    p.add_argument('--source-deployment-sha');p.add_argument('--source-summary-sha')
    a=p.parse_args();selfcheck()
    if a.input:
        digest=hashlib.sha256(a.input.read_bytes()).hexdigest();assert digest==a.sha256 and a.out and a.prereg
        data=json.loads(a.input.read_text());assert data['prereg']==a.prereg
        source_receipts={}
        if 'units' not in data:
            assert a.source_deployment197 and a.source_summary197 and a.source_deployment_sha and a.source_summary_sha
            for label,path,expected in (('deployment197',a.source_deployment197,a.source_deployment_sha),('summary197',a.source_summary197,a.source_summary_sha)):
                actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==expected
                source_receipts[label]=actual
            data=normalize_peer(data,json.loads(a.source_deployment197.read_text()),json.loads(a.source_summary197.read_text()))
        callback=None
        if not a.phase1:
            if a.graph_adapter is None:p.error('F1 requires--graph-adapter;or explicit--phase1')
            spec=importlib.util.spec_from_file_location('endpoint198_graph_adapter',a.graph_adapter)
            adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
            callback=adapter.make_scorer(data)
        result=run_units(data['units'],callback);result['input_sha256']=digest;result['prereg']=a.prereg
        result['source197_receipts']=source_receipts
        a.out.mkdir(parents=True,exist_ok=False);path=a.out/'capacity.json'
        path.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')
        (a.out/'SEALED.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
        print(json.dumps(result['hives'],indent=2))
