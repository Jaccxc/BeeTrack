"""Project sealed §214 stages into a small anonymous numeric replay bundle.

This packer is intentionally the only place that reads the private producer
trees.  It strips paths, frame-row IDs, GT IDs, coordinates, crops and member
names before writing evidence214.json.gz.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

DOMAINS=('35','33','W','G')
ARMS=('K','G','I','Z','O')
EXPECTED={
    'frozen':'ba6d026f3026f78d6809779d12b171588460abac5ce6a1065343566d6b0794fa',
    'labels':'732ee647f22ecf77d6ca227b6a760523a24dc155cc5f7191dc4f39b5d441804f',
    'predictions':'c71ddcc0188fcb87cb37ffec8a05990d78a1ffbb8c005d1eeab8caff9fac5569',
    'score':'31cba8bdba2dc6b36a6ad735259d0d80ff15a154759e4e8171cb6bd1c2702009',
    'verification':'c984d268bcf36fa1228ced8d4e5e7eb3e12afff40e830a2e2341b1f4da4d65cc',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load_json(path):return json.loads(Path(path).read_text(encoding='utf8'))


def load_gz(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)


def verify_tree(root,expected):
    root=Path(root)
    seal=(root/'SEALED.sha256').read_text().strip()
    if seal!=expected or sha(root/'manifest.json')!=seal:raise ValueError('stage top seal mismatch: '+root.name)
    files=load_json(root/'manifest.json')
    for name,digest in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('stage file changed: '+root.name+'/'+name)
    return seal


def opaque_maps(roster,clip_id):
    endpoint_map={};run_map={};run_order={};features={}
    for slot in (0,1,2):
        records=sorted((r for r in roster['records'] if r['native_slot']==slot),key=lambda r:int(r['raw_index']))
        canon=sorted({str(r['run_id']) for r in records})
        # String ordering preserves the source fitter's canonical-run tie/order convention.
        run_map[slot]={old:f'r{i+1:04d}' for i,old in enumerate(canon)}
        run_order[slot]={old:i for i,old in enumerate(canon)}
        for i,r in enumerate(records):
            endpoint_map[r['endpoint_id']]=f'{clip_id}/m{slot}/e{i+1:05d}'
            fe=r.get('features') or {}
            row=fe.get('feature_row')
            features[r['endpoint_id']]=dict(endpoint_id=endpoint_map[r['endpoint_id']],member_index=slot,
                run_id=run_map[slot][str(r['run_id'])],run_order=run_order[slot][str(r['run_id'])],
                endpoint_order=i,eligible=bool(r['eligible']),reason=r.get('gate',{}).get('reason'),
                feature_row=row,raw_index=int(r['raw_index']))
    return endpoint_map,features


def opaque_state(score,endpoint_map):
    metric=score['metrics'];ev=metric['evidence']
    pred_raw=ev['predictions'];gt_raw=ev['gt']
    pmap={};gmap={}
    pred=[]
    for row in pred_raw:
        key=int(row[0]);pmap.setdefault(key,f'p{len(pmap)+1:04d}')
        pred.append([pmap[key],int(row[1])])
    gt=[]
    for row in gt_raw:
        key=int(row[0]);gmap.setdefault(key,f'g{len(gmap)+1:04d}')
        gt.append([gmap[key],int(row[1])])
    graphs={}
    for t,key in (('0.5','edges05'),('0.3','edges03')):
        graphs[t]={'edges':[[pmap[int(p)],gmap[int(g)]] for p,g in ev[key]]}
    diag=[]
    for r in score['endpoint_diagnostics']['records']:
        original=r['endpoint_id']
        if original not in endpoint_map:raise ValueError('diagnostic endpoint missing from frozen roster')
        keep=('original_matched','moved','original_side','corrected_side','side_flipped',
              'original_signed_polygon_distance','corrected_signed_polygon_distance','original_box_diagonal',
              'GT_side','GT_signed_polygon_distance','before_boundary_MAE','after_boundary_MAE',
              'before_center_error','after_center_error','before_side_error','after_side_error',
              'flip_repaired','flip_damaged')
        diag.append(dict(endpoint_id=endpoint_map[original],**{k:r[k] for k in keep if k in r}))
    counts=score['counts']
    declared=dict(numerator=int(counts['numerator']),denominator=int(counts['denominator']),
        W=float(counts['W']),nGT=int(counts['nGT']),nPred=int(counts['nPred']),iou={})
    for t in ('0.5','0.3'):
        q=counts['iou'][t];declared['iou'][t]={k:q[k] for k in ('TP','Pred','GT','FP','FN','E','F1')}
    return dict(pred_events=pred,gt_events=gt,graphs=graphs,
        bins=dict(pred=metric['pred_bins'],gt=metric['gt_bins']),declared=declared,
        endpoint_diagnostics=diag)


def project(args):
    roots={k:Path(getattr(args,k)) for k in EXPECTED}
    seals={k:verify_tree(v,EXPECTED[k]) for k,v in roots.items()}
    independent=load_json(roots['verification']/'independent.json')
    if not independent.get('passed') or independent['input_seals']!={k:v for k,v in seals.items() if k!='verification'}:
        raise ValueError('independent audit receipt does not bind all four stage seals')
    dep=load_json(roots['frozen']/'deployment.json')
    pred=load_json(roots['predictions']/'predictions.json')
    score_summary=load_json(roots['score']/'summary.json')
    if pred['schema']!='center214-source-ridge-v1' or score_summary['schema']!='center214-native-evaluation-v1':raise ValueError('unknown source/evaluation schemas')
    prediction_by={(m['domain'],m['clip'],int(m['member_index'])):m for m in pred['members']}
    eval_clip={(c['domain'],c['clip']):c for c in score_summary['clips']}
    if len(eval_clip)!=6 or len(prediction_by)!=18:raise ValueError('expected fixed six clips and 18 members')

    clip_names={h:sorted(c['clip'] for c in dep['clips'] if c['domain']==h) for h in DOMAINS}
    clip_alias={(h,c):f'c{i}' for h,names in clip_names.items() for i,c in enumerate(names)}
    roster_data={};endpoint_map={};endpoint_info={};run_aliases={}
    target_rows=[];member_rows={};source_by_original={}
    for ref in dep['clips']:
        h,c=ref['domain'],ref['clip'];folder=roots['frozen']/ref['path']
        roster=load_gz(folder/'roster.json.gz');feature_names=load_json(folder/'feature_rows.json')
        with np.load(folder/'data.npz',allow_pickle=False) as z:G=z['G'];CHI=z['chi']
        if len(feature_names)!=len(G) or G.shape!=(len(CHI),6) or CHI.shape[1:]!=(24,):raise ValueError('feature bank row mapping differs')
        feature_index={old:i for i,old in enumerate(feature_names)}
        cid=f'{h}/{clip_alias[h,c]}'
        emap,fmap=opaque_maps(roster,cid);endpoint_map.update(emap);endpoint_info.update(fmap)
        for slot in range(3):
            predmember=prediction_by[h,c,slot]
            if len(predmember['endpoint_ids'])!=sum(r['native_slot']==slot for r in roster['records']):raise ValueError('prediction/frozen endpoint roster length differs')
            if set(predmember['endpoint_ids'])!={r['endpoint_id'] for r in roster['records'] if r['native_slot']==slot}:raise ValueError('prediction/frozen endpoint inventory differs')
            pos={x:i for i,x in enumerate(predmember['endpoint_ids'])}
            shift_values={a:np.load(roots['predictions']/predmember['shifts'][a],allow_pickle=False) for a in ('K','G','I','Z')}
            if any(v.shape!=(len(pos),2) for v in shift_values.values()):raise ValueError('certified shift shape differs')
            for rec in sorted((r for r in roster['records'] if r['native_slot']==slot),key=lambda r:int(r['raw_index'])):
                info=fmap[rec['endpoint_id']];ix=feature_index.get(rec['endpoint_id'])
                eligible=bool(rec['eligible'])
                if eligible!=(ix is not None) or eligible!=(info['feature_row'] is not None) or (eligible and int(info['feature_row'])!=ix):raise ValueError('frozen feature eligibility/bank mismatch')
                row=dict(endpoint_id=info['endpoint_id'],domain=h,clip=clip_alias[h,c],member_index=slot,
                    member=f'm{slot}',run_id=info['run_id'],run_order=info['run_order'],endpoint_order=info['endpoint_order'],
                    eligible=eligible,reason=info['reason'],G6=G[ix].tolist() if eligible else None,
                    chi24=CHI[ix].tolist() if eligible else None,
                    certified_shifts={a:shift_values[a][pos[rec['endpoint_id']]].tolist() for a in ('K','G','I','Z')})
                target_rows.append(row);member_rows.setdefault((h,c,slot),[]).append(row)
                source_by_original[rec['endpoint_id']]=row
        roster_data[h,c]=roster

    # Convert authoritative fitted training records into anonymous source rows
    # and preserve each fold's exact training sequence without raw indices.
    source_rows_by_id={};fold_orders={};model_refs={}
    for heldout in DOMAINS:
        model=load_json(roots['predictions']/'models'/f'{heldout}.json')
        if model['heldout']!=heldout:raise ValueError('wrong heldout source model')
        order={}
        for i,r in enumerate(model['training_records']):
            if r['endpoint_id'] not in endpoint_map:raise ValueError('fit row not in anonymous endpoint roster')
            anon=endpoint_map[r['endpoint_id']]
            target=source_by_original[r['endpoint_id']]
            if not target['eligible'] or target['domain']==heldout or target['member_index']!=0:raise ValueError('forbidden source/holdout endpoint in fit row')
            row=dict(endpoint_id=anon,domain=target['domain'],clip=target['clip'],member='m0',member_index=0,
                role='receiverK',run_id=target['run_id'],run_order=target['run_order'],endpoint_order=target['endpoint_order'],
                eligible=True,target_known=True,G6=target['G6'],chi24=target['chi24'],
                target_clipped=list(map(float,r['target'])))
            previous=source_rows_by_id.get(anon)
            if previous is not None:
                for k in ('domain','clip','run_id','G6','chi24','target_clipped'):
                    if previous[k]!=row[k]:raise ValueError('fold training row differs across source models')
            source_rows_by_id[anon]=row;order[anon]=i
        fold_orders[heldout]=order
        model_refs[heldout]={}
        for head in ('G','I'):
            model_refs[heldout][head]={k:model['heads'][head][k] for k in ('status','mean','scale','beta','intercept','zero_scale_columns')}
    source_rows=list(source_rows_by_id.values())

    clips=[]
    for h,c in sorted(eval_clip,key=lambda x:(DOMAINS.index(x[0]),clip_alias[x])):
        source_clip=eval_clip[h,c];roster=roster_data[h,c]
        members=[]
        for slot in (0,1,2):
            state_refs={}
            for arm in ARMS:
                path=roots['score']/'clips'/h/c/f'm{slot}'/arm/'score.json'
                state=load_json(path)
                if (state['domain'],state['clip'],int(state['member_index']),state['arm'])!=(h,c,slot,arm):raise ValueError('score state path identity mismatch')
                state_refs[arm]=opaque_state(state,endpoint_map)
            members.append(dict(member=f'm{slot}',member_index=slot,nframes=int(source_clip['nf']),
                endpoint_ids=[r['endpoint_id'] for r in member_rows[h,c,slot]],
                k212_closure=source_clip['members'][slot]['K212_closure'],states=state_refs))
        capacity=load_json(roots['score']/'capacity'/h/(c+'.json'))
        bound=capacity['replay_evidence']
        if len(bound['members'])!=3:raise ValueError('bound sidecar member roster differs')
        clips.append(dict(domain=h,clip=clip_alias[h,c],nframes=int(source_clip['nf']),members=members,
            bound_evidence=bound))

    verification=load_json(roots['verification']/'physical.json')
    member_alias={}
    for h in DOMAINS:
        sample=next(c for c in score_summary['clips'] if c['domain']==h)
        member_alias[h]={str(m['member']):int(m['member_index']) for m in sample['members']}
        if set(member_alias[h].values())!={0,1,2}:raise ValueError('domain member roster cannot be anonymized')
    bounds_reference={}
    for h in DOMAINS:
        row=verification['count_bounds'][h]
        bounds_reference[h]=dict(E3=row['E3'],members=[dict(member_slot=member_alias[h][str(m['member'])],
            denominator=int(m['denominator']),lower_bound_numerators={k:int(v) for k,v in m['lower_bound_numerators'].items()})
            for m in row['members']])
        if len(bounds_reference[h]['members'])!=3:raise ValueError('independent member bound summary incomplete')
    evidence=dict(schema='center214.numeric.v1',prereg='79dbed053bcddd4290300494ceff29c8130a5807',
        supplement='21c6c803290fad2816ca472ec673b9f2eda76759',receipts=seals,
        scope=dict(original_K_members=18,endpoint_rows=len(target_rows),eligible_endpoints=sum(r['eligible'] for r in target_rows),
            source_rows=len(source_rows),native_arm_scores=90,arm_names=list(ARMS),
            image_chi_extraction='independent original-image/G6/chi24 audit only; raw crops and pixels omitted',
            GT_assignment='independent full-scene original-row owner audit only; raw GT IDs/coordinates omitted',
            changed_arm='fresh official two-IoU graphs and 30-offset windows are replayed numerically from anonymized vertices/edges'),
        source_rows=source_rows,fold_orders=fold_orders,model_references=model_refs,
        endpoint_rows=target_rows,clips=clips,independent_count_bounds=bounds_reference,
        independent_gate_references=dict(primary=verification['primary_checks'],
            crop_spatial_information=verification['mechanism']))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('immutable numeric evidence path already exists')
    with out.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=9) as gz:
            gz.write(json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8'))
    return dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size,
                endpoints=len(target_rows),eligible=sum(r['eligible'] for r in target_rows),source_rows=len(source_rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.frozen=a.root/'frozen';a.labels=a.root/'labels';a.predictions=a.root/'predictions';a.score=a.root/'score';a.verification=a.root/'verification'
    print(json.dumps(project(a),sort_keys=True))


if __name__=='__main__':main()
