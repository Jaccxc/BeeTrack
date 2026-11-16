"""Replay §117 full-tail event scores and temporal-donor evidence without loading video or GT tracks."""
import argparse
from fractions import Fraction
from base64 import b64decode, b64encode
import hashlib
import json
import lzma
import math
from pathlib import Path
from statistics import median
import sys

# The public checkout already contains the §115 independent sparse matcher and window implementation.
project = Path(__file__).resolve().parents[2]
if (project / "tools/study115/graph115.py").is_file():
    sys.path.insert(0, str(project))
from tools.study115.graph115 import Graph, _events, _grids, _window
from tools.study115.release115 import numerical, read, require, save, wire


ARMS = ("U", "A", "Z", "Q", "Zm", "D")
TAUS = {"0.5":"edges05", "0.3":"edges03"}
PREREG = '064d5bf1786d0a9733ec7c99496e46679548fb3f'
COORDINATE_NOTE = '027e90968e67e4a86dbfd462b4f941efdc610934'


def object_sha(value):
    return hashlib.sha256(wire(value)).hexdigest()


def task_scores(evidence):
    n = evidence["nframes"]
    require(n == 1000, "Expected a complete 1000-frame source tail")
    truth = _events(evidence["gt"], n)
    pred = _events([row[:2] for row in evidence["predictions"]], n)
    counts = lambda records: [sum(t > 0 for t in records.values()), sum(t < 0 for t in records.values())]
    gt_counts, pred_counts = counts(truth), counts(pred)
    w, grids = _window(_grids(pred.values(), n), _grids(truth.values(), n), 1, gt_counts)
    result = dict(w30_percent=w["w_percent"], gt_counts=dict(zip(("in","out"),gt_counts)),
                  pred_counts=dict(zip(("in","out"),pred_counts)), bias=dict(zip(("in","out"),w["bias_in_out"])),
                  window=dict(numerator=sum(map(sum,grids["abs_error_in_out_by_offset"])),
                              denominator=30*len(truth), l1_by_offset_in_out=grids["abs_error_in_out_by_offset"]),
                  iou={})
    for tau, key in TAUS.items():
        score = Graph(pred, truth, evidence[key]).score()
        for d in score["by_direction"].values():
            d["f1"] = 2*d["tp"]/max(1,2*d["tp"] + d["fp"] + d["fn"])
        result["iou"][tau] = score
    return result


def check_task(score):
    e = score["evidence"]
    require(object_sha(e) == score["evidence_sha256"], "Official event graph fingerprint changed")
    actual = task_scores(e)
    for key in ("gt_counts", "pred_counts", "bias"):
        require(score[key] == actual[key], "Official directional count differs")
    aw, ew = actual["w30_percent"], score["w30_percent"]
    require(aw == ew or (aw is not None and ew is not None and math.isclose(aw, ew, abs_tol=1e-12)),
            "Official window value differs")
    require(all(val == score["window"][key] for key, val in actual["window"].items()), "Window evidence differs")
    for tau, r in actual["iou"].items():
        original = score["iou"][tau]
        require(all(v == original[k] for k,v in r.items()), "Sparse maximum matching differs from official result")
        ids = original["matching"]
        legal = set(map(tuple, e[TAUS[tau]]))
        require(len(ids) == len({p for p,_ in ids}) == len({g for _,g in ids}) == r["tp"] and
                all(tuple(x) in legal for x in ids), "Official matching does not realize the reported maximum")
    return actual


def intern(pool, lookup, value):
    key = object_sha(value)
    if key not in lookup:
        lookup[key] = len(pool)
        pool.append(value)
    return lookup[key]


def public_score_record(score, pool, lookup):
    results = check_task(score)
    e = score["evidence"]
    # Identical arms share a graph even when the bytes of their MOT input differ elsewhere in the clip.
    body = {k:e[k] for k in ("gt", "predictions", *TAUS.values())}
    return dict(graph=intern(pool,lookup,body), graph_sha256=score["evidence_sha256"],
                inputs_sha256=e["inputs_sha256"], results_sha256=object_sha(results),
                matching={t:score['iou'][t]['matching'] for t in TAUS})


def reconstruct_record(record, pool):
    require(type(record["graph"]) is int and 0 <= record["graph"] < len(pool), "Invalid public graph reference")
    e = dict(format="temporal117-official-graph-v1", nframes=1000,
             inputs_sha256=record["inputs_sha256"], **pool[record["graph"]])
    require(object_sha(e) == record["graph_sha256"], "Official archived graph does not reproduce")
    actual = task_scores(e)
    require(object_sha(actual) == record["results_sha256"], "Reconstruction differs from the scored archive")
    return actual


def fold_result(clip, cases):
    """Compute the registered full-P0 task and added-complexity gates as rational contrasts."""
    values = {}
    for s in range(3):
        for arm in ARMS:
            r = cases[str(s)][arm]
            w = r['window']
            metrics = {'w': Fraction(100*w['numerator'], w['denominator'])}
            for tau, v in r['iou'].items():
                metrics[tau] = Fraction(2*v['tp'], max(1,2*v['tp']+v['fp']+v['fn']))
            values[s,arm] = metrics
    average = {arm:{key:sum((values[s,arm][key] for s in range(3)),Fraction())/3 for key in ('w',*TAUS)}
               for arm in ARMS}
    means = {arm:dict(w=float(m['w']),f1={t:float(m[t]) for t in TAUS}) for arm,m in average.items()}
    deltas = []
    for s in range(3):
        u = values[s,'U']
        for arm in ARMS:
            m = values[s,arm]
            p = cases[str(s)]
            deltas.append(dict(seed=s, arm=arm,w_reduction_pp=float(u['w']-m['w']),
                               f1_delta={t:float(m[t]-u[t]) for t in TAUS},
                               direction_count_delta={d:p[arm]['pred_counts'][d]-p['U']['pred_counts'][d]
                                                      for d in ('in','out')}))
    eps = Fraction(1,10**9)
    a = average['A']
    gain = average['U']['w']-a['w']
    delta = a['0.5']-average['U']['0.5']
    wins = sum(values[s,'U']['w']-values[s,'A']['w'] > eps for s in range(3))
    controls = {arm:dict(w_advantage_pp=float(average[arm]['w']-a['w']),
                         f1_delta=float(a['0.5']-average[arm]['0.5'])) for arm in ('Z','Q')}
    complex_pass = all(average[c]['w']-a['w'] > eps and a['0.5']-average[c]['0.5'] >= -eps for c in ('Z','Q'))
    return dict(clip=clip,means=means,against_new_cpu_u=deltas,
                utility=dict(mean_w_gain_pp=float(gain),positive_seeds=wins,mean_f1_delta=float(delta),
                             passed=gain>=2 and wins>=2 and delta>=-eps),
                complexity=dict(comparisons=controls,passed=complex_pass))


def archive_fragment(value):
    """Compress one case before loading its neighbor. Binary64 values retain all original digits."""
    data = lzma.compress(wire(numerical(value)), preset=6)
    return b64encode(data).decode('ascii')


def unpack_fragment(data):
    return numerical(json.loads(lzma.decompress(b64decode(data,validate=True))), decode=True)


def schedule():
    return [(fold, s, a) for fold in (33,35) for s in range(3) for a in ARMS]


def add_e2e(pack, root, score_hash, archive):
    from history117 import history
    scores = read(root/'scores/index.json',score_hash)
    tracking = read(root/'tracking/index.json',scores['tracking_index_sha256'])
    detector = read(root/'detector_done.json',tracking['detector_index_sha256'])
    official = read(root/'scores'/scores['summary_file'],scores['summary_sha256'])
    require(pack['prereg'] == scores['prereg'] == tracking['prereg'] == detector['prereg'] and
            tracking['stage'] == 'predictions_only', 'Tracking/scoring freeze chain mismatch')
    pack['source_sha256'].update(scores=score_hash, tracking=scores['tracking_index_sha256'],
                                detector=tracking['detector_index_sha256'], summary=scores['summary_sha256'])
    pack['indices'] = dict(detector=detector, tracking=tracking, scores=scores)
    for index in (tracking,scores):
        require([(r['fold'],r['seed'],r['arm']) for r in index['cases']] == schedule(),
                'Incomplete 36-condition source sample')
    lookup = {}
    aggregate_rows = {}
    for tr, sr in zip(tracking['cases'], scores['cases']):
        fold,seed,arm,clip = (tr[k] for k in ('fold','seed','arm','clip'))
        s = str(seed)
        name = f'{fold}_{seed}'
        if name not in pack['acquisition']:
            rel = f'cases/{name}.json'
            pack['acquisition'][name] = read(root/rel,detector['files'][rel])
        cnt = read(root/'tracking'/tr['file'],tr['sha256'])
        p = read(root/'scores'/sr['file'],sr['sha256'])
        require(p['mot_sha256'] == tr['mot_sha256'] == cnt['mot_sha256'] and
                cnt['det_sha256'] == detector['files'][f'detections/{name}/{arm}.npy'] and
                [[i,t,*[round(x,2) for x in b]] for i,t,*b in p['evidence']['predictions']] == cnt['predictions'],
                'Scored MOT and pre-GT prediction freeze disagree')
        row = public_score_record(p,pack['graphs'],lookup)
        row['gt_file_sha256'] = p['gt_sha256']
        row['tracking_receipt'] = cnt['tracking_receipt']
        pack['cases'].setdefault(clip,{}).setdefault(s,{})[arm] = row
        aggregate_rows.setdefault(clip,{}).setdefault(s,{})[arm] = reconstruct_record(row,pack['graphs'])
    for fold in (33,35):
        clip = f'BEE24-{68-fold}-tail'
        require(fold_result(clip,aggregate_rows[clip]) == official['folds'][str(fold)],
                'Independent task/complexity gate differs from official fold report')
    hist = history(archive)
    require(hist['archive_sha256'] == official['historical_gpu_fp16']['archive_sha256'],
            'Different historical source archive')
    pack['source_sha256']['history'] = hist['archive_sha256']
    for clip, seeds in hist['cases'].items():
        for s, r in seeds.items():
            original = official['historical_gpu_fp16']['cases'][f"{r['fold']}_{s}_tail"]
            require(r['pred_counts'] == original['pred_counts'] and r['gt_counts'] == original['gt_counts'] and
                    abs(r['w30_percent']-original['w30_percent']) < 1e-12 and
                    all(r['iou'][t]['tp'] == original['iou'][t]['tp'] for t in TAUS),
                    'Historical control differs from official score')
            e = r['evidence']
            pack['history'].setdefault(clip,{})[s] = dict(graph=intern(pack['graphs'],lookup,{k:e[k] for k in
                ('gt','predictions',*TAUS.values())}),provenance={k:v for k,v in r.items() if k.endswith('_sha256')})


def add_baselines(pack, root, expected):
    from baselines117 import report
    b = read(root/'baselines/replay.json.gz',expected)
    require(b['prereg'] == pack['prereg'] and b['primary_index_sha256'] == pack['source_sha256']['tracking'] and
            report(b['records']) == b['results'], 'Tracker comparisons derived from a different freeze')
    sanitized = {}
    for key, row in b['records'].items():
        info = {k:v for k,v in row['info'].items() if k.endswith('sha256') or k in (
                'fold','seed','nframes','clip','topic_tracker','calibration','detector_comparison')}
        sanitized[key] = {'info':info,'bytetrack':row['bytetrack'],'topic':row['topic']}
    require(report(sanitized) == b['results'], 'Compacted baseline proofs fail to reproduce')
    pack['baselines'] = dict(records=sanitized, reported_sha256=object_sha(b['results']))
    pack['source_sha256'].update(baselines=expected, baselines_index=b['baseline_index_sha256'])


def add_mechanism(bundle, root, index_sha):
    from donors import replay_mechanism
    directory = root/'mechanism'
    index = read(directory/'index.json',index_sha)
    require(index['prereg'] == bundle['prereg'] and index['detector_index_sha256'] == bundle['source_sha256']['detector']
            and index['tracking_index_sha256'] == bundle['source_sha256']['tracking'], 'Mechanism seal mismatch')
    bundle['indices']['mechanism'] = index
    bundle['source_sha256']['mechanism'] = index_sha
    for fold in (33,35):
        for seed in range(3):
            key = f'{fold}_{seed}'
            source = index['cases'][key]
            file_digest = source['sha256']
            data = read(directory/source['file'],file_digest)
            meta = data['meta']
            require((meta['fold'],meta['seed'],meta['clip'],len(data['frames'])) ==
                    (fold,seed,f'BEE24-{68-fold}-tail',1000), 'Unexpected mechanism sample')
            result = replay_mechanism(data)
            ref_sha = source['summary_sha256']
            require(read(directory/source['summary_file'],ref_sha) == result, 'Official mechanism result differs')
            bundle['mechanism'][key] = dict(source_sha256=file_digest,summary_sha256=ref_sha,
                evidence_sha256=object_sha(data),proof=archive_fragment(data),result_sha256=object_sha(result))


def observed_costs(data, samples):
    """Retain nonoverlapping and nested timings separately. Each mean uses measured frame values."""
    times = {}
    counts = {}
    maxima = {}
    total = 0
    def collect(current, prefix=''):
        for k,v in current.items():
            name = prefix+k
            if isinstance(v,dict):
                collect(v,name+'.')
            elif type(v) in (float,int):
                require(math.isfinite(v) and v>=0, 'Invalid measured duration')
                times[name] = times.get(name,0.0) + v
                counts[name] = counts.get(name,0) + 1
    for row in data['frames']:
        r = row['runtime']
        require(r['backbone_forwards']==1 and r['stock_u_exact'] is True, 'Native forward receipt mismatch')
        total += 1
        t = r['timing_ms']
        collect(t)
        nms=t['nms_detail']
        u=nms['stock_reference']
        a=nms['indexed']['A']['standalone_proxy_ms']
        shared=t['read_hash_decode']+t['preprocess']+t['forward']
        values=dict(decode=t['read_hash_decode'],preprocess=t['preprocess'],forward=t['forward'],
                    features_a_prefix=t['a_path_ms'],features_six=t['features_six'],reference_nms=u,
                    a_nms_proxy=a,compute_six=t['compute_total'],u_component_sum=shared+u,
                    a_component_sum=shared+t['a_path_ms']+a)
        for k,v in values.items():
            samples.setdefault(k,[]).append(v)
        for k in ('process_rss_bytes','process_peak_rss_bytes','history_native_bytes','history_diagnostic_boxes_bytes'):
            if r.get(k) is not None:
                maxima[k]=max(maxima.get(k,0),r[k])
    return dict(backbone_forwards=total, max_bytes=maxima,
                mean_ms={key:v/counts[key] for key,v in times.items()}, n_measurements=counts)


def verify_donor_rows(data):
    frames = data['frames']
    require(data['owner_geometry']=='unclipped_primary_clipped_sensitivity','Incorrect anchor-ownership convention')
    for f, row in enumerate(frames,1):
        gt, pools, eligible = set(row['gt']), row['pools'], row['eligible']
        require(f==row['f'] and row['donor_frame']==(f-2 if f>=3 else None) and len(gt)==len(row['gt']),
                'Invalid frame clock or GT roster')
        require(set(row['arms'])==set(ARMS) and set(row['arms'].values())==set(range(len(pools))),
                'Invalid or unused frame arm')
        for p in pools:
            ids = set(p['ids'])
            matches = p['matches']
            require(len(ids)==len(p['ids']) and len(matches)==len({x[0] for x in matches})==len({x[1] for x in matches})
                    and all(g in gt and a in ids and .3<=iou<=1 and type(side) is bool for g,a,iou,side in matches),
                    'Invalid Hungarian ownership witness')
        for i,v in enumerate(eligible):
            require(f>=3 and v['D_donor']==eligible[i-1]['donor'] and .05<=v['scores']['U']<.25<=v['donor_score'],
                    'Incorrect raw eligibility or displaced-donor control')
            old = set(frames[f-3]['gt'])
            for name,truth in (('current_top2',gt),('current_top2_clipped',gt),('donor_top2',old),('donor_top2_clipped',old)):
                pairs = v[name]
                require(len(pairs)==min(2,len(truth)) and pairs==sorted(pairs,key=lambda p:(-p[1],p[0]))
                        and len(pairs)==len({g for g,_ in pairs}) and all(g in truth and 0<=q<=1 for g,q in pairs),
                        'Incorrect top/runner-up donor overlap')


def validate_indices(bundle):
    idx = bundle['indices']
    require(bundle['prereg']==PREREG and all(object_sha(idx[k]) == bundle['source_sha256'][k]
                and idx[k]['prereg']==PREREG for k in ('scores','tracking','detector','mechanism')),
            'Official parent indices do not reproduce')
    for key,r in bundle['acquisition'].items():
        fold,s = map(int,key.split('_'))
        conf = r['config']
        require(object_sha(r) == idx['detector']['files'][f'cases/{key}.json'] and
                (conf['fold'],conf['seed'],conf['clip'],conf['prereg']) == (fold,s,f'BEE24-{68-fold}-tail',PREREG)
                and r['backbone_forwards']==r['frames']==1000, 'Acquisition receipt differs')
    require(len(bundle['acquisition'])==6 and idx['detector']['backbone_forwards']==6000, 'Incomplete inference')
    for r, sr in zip(idx['tracking']['cases'],idx['scores']['cases']):
        entry = bundle['cases'][r['clip']][str(r['seed'])][r['arm']]
        receipt = entry['tracking_receipt']
        key=f"{r['fold']}_{r['seed']}"
        require(receipt['mot_sha256']==r['mot_sha256'] and receipt['input_file_sha256']==
                idx['detector']['files'][f"detections/{key}/{r['arm']}.npy"], 'MOT provenance differs')
        graph = bundle['graphs'][entry['graph']]
        ids = {k:r[k] for k in ('clip','fold','seed','arm')}
        e = dict(format='temporal117-official-graph-v1',nframes=1000,inputs_sha256=entry['inputs_sha256'],**graph)
        actual = task_scores(e)
        grids = _grids([t for _,t,*_ in e['predictions']],1000)
        max_bins = max(map(len,grids))
        counts = dict(**ids,nframes=1000,prereg=PREREG,detector_index_sha256=bundle['source_sha256']['detector'],
                      det_sha256=receipt['input_file_sha256'],zone_sha256=idx['tracking']['zone_sha256'],
                      mot_file=r['mot_file'],mot_sha256=r['mot_sha256'],method_sha256=idx['tracking']['method_sha256'],
                      tracking_receipt=receipt,totals=actual['pred_counts'],
                      windows=[cells+[[0,0]]*(max_bins-len(cells)) for cells in grids],
                      predictions=[[pid,t,*[round(x,2) for x in box]] for pid,t,*box in e['predictions']])
        require(object_sha(counts)==r['sha256'], 'Pre-GT count file does not reproduce')
        actual['window'].update(width_frames=750,fps=25,offsets_frames=list(range(0,750,25)),
                                bins_per_offset=list(map(len,grids)))
        actual['bias']={k:int(v) for k,v in actual['bias'].items()}
        for t in TAUS:
            actual['iou'][t]['matching'] = entry['matching'][t]
        original = dict(**ids,nframes=1000,mot_sha256=r['mot_sha256'],gt_sha256=entry['gt_file_sha256'],
                        evidence=e,evidence_sha256=entry['graph_sha256'],**actual)
        require(object_sha(original)==sr['sha256'], 'Complete official score file does not reproduce')


def replay(bundle):
    from baselines117 import report as baseline_report
    from donors import replay_mechanism, aggregate
    require(bundle['format']=='beetrack-study117-v1' and bundle['coordinate_note']==COORDINATE_NOTE, 'Unknown experiment')
    validate_indices(bundle)
    report = dict(cases={}, folds={}, mechanism={}, acquired={}, costs={}, cost_by_fold={}, historical={})
    for clip,seeds in bundle['cases'].items():
        fold = 68-int(clip.split('-')[1])
        mechanism = {}
        samples = {}
        for s,arms in seeds.items():
            require(set(arms)==set(ARMS) and len(seeds)==3, 'Incomplete arm comparison')
            key = f'{fold}_{s}'
            scored = {arm:reconstruct_record(r,bundle['graphs']) for arm,r in arms.items()}
            report['cases'].setdefault(clip,{})[s] = scored
            h = bundle['history'][clip][s]
            hist = dict(nframes=1000, **bundle['graphs'][h['graph']])
            report['historical'].setdefault(clip,{})[s]=dict(scores=task_scores(hist),provenance=h['provenance'])
            m = bundle['mechanism'][key]
            data = unpack_fragment(m['proof'])
            require(object_sha(data)==m['evidence_sha256'], 'Donor evidence was changed')
            verify_donor_rows(data)
            mechanism[s] = replay_mechanism(data)
            require(object_sha(mechanism[s])==m['result_sha256'], 'Donor replay differs from registered outcomes')
            if 'native_frame_agreement' in bundle.get('validation',{}):
                exact=bundle['validation']['native_frame_agreement']['data']['cases'][key]
                require(all(exact[a][t]==mechanism[s]['thresholds'][t]['frame_counts'][a] for a in ARMS for t in TAUS),
                        'Portable detection counts differ from physical full-frame matches')
            require(data['meta']['sqlite_sha256']==bundle['indices']['detector']['files'][f'cases/{key}.sqlite']
                    and data['meta']['source_gt_sha256']==arms['U']['gt_file_sha256'], 'Wrong donor source')
            cost = observed_costs(data,samples)
            cost['tracking']={a:r['tracking_receipt'] for a,r in arms.items()}
            report['costs'].setdefault(clip,{})[s] = cost
            report['acquired'][key]=bundle['acquisition'][key]
        report['folds'][str(fold)] = fold_result(clip,report['cases'][clip])
        by_seed = report['costs'][clip]
        hist = {(x['max_bytes']['history_native_bytes'],x['max_bytes']['history_diagnostic_boxes_bytes'])
                for x in by_seed.values()}
        require(len(hist)==1 and len(samples['forward'])==3000, 'Incomplete full-source cost comparison')
        native,audit=hist.pop()
        report['cost_by_fold'][str(fold)]=dict(frames=3000,medians_ms={k:round(median(v),4) for k,v in samples.items()},
                                              two_frame_feature_logit_bytes=native,diagnostic_box_bytes=audit)
        report['mechanism'][clip] = dict(cases=mechanism,across_seeds=aggregate(list(mechanism.values())))
        require(report['mechanism'][clip]['across_seeds']==bundle['indices']['mechanism']['folds'][str(fold)],
                'Cross-seed donor gate differs from source report')
    b = bundle['baselines']
    report['comparators'] = baseline_report(b['records'])
    require(object_sha(report['comparators']) == b['reported_sha256'], 'Comparator report differs')
    report['advance_to_unseen'] = all(f['utility']['passed'] and f['complexity']['passed']
                                      for f in report['folds'].values())
    if 'cost117' in bundle.get('validation',{}):
        require(report['cost_by_fold'] == bundle['validation']['cost117']['data'], 'Original measured cost report differs')
    if 'report_sha256' in bundle:
        require(object_sha(report)==bundle['report_sha256'], 'Reconstructed full report differs')
    return report


def print_table(output):
    print('source seed arm      W%    F1@0.5   F1@0.3    in   out')
    for clip,seeds in sorted(output['cases'].items()):
        for seed, arms in sorted(seeds.items()):
            for arm in ARMS:
                score = arms[arm]
                print(f"{clip} {seed} {arm:3} {score['w30_percent']:8.3f} {score['iou']['0.5']['f1']:8.4f} "
                      f"{score['iou']['0.3']['f1']:8.4f} {score['pred_counts']['in']:5} {score['pred_counts']['out']:5}")
        f=output['folds'][str(68-int(clip.split('-')[1]))]
        m=output['mechanism'][clip]['across_seeds']
        print(clip,'utility',f['utility']['passed'],'complexity',f['complexity']['passed'],
              'mechanism',m['source_pass'],'passing seeds',m['passed_seeds'])


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest='command',required=True)
    make=sub.add_parser('pack')
    make.add_argument('root',type=Path)
    make.add_argument('output',type=Path)
    make.add_argument('--history',type=Path,default=Path('docs/study108_source.json.gz'))
    for option in ('score','mechanism','baseline'):
        make.add_argument('--'+option+'-sha',required=True)
    view=sub.add_parser('replay')
    view.add_argument('archive',type=Path)
    view.add_argument('--report',type=Path)
    view.add_argument('--table',action='store_true')
    witness=sub.add_parser('evidence',help='Export one complete 1000-frame donor/ownership proof as numeric JSON')
    witness.add_argument('archive',type=Path)
    witness.add_argument('fold',type=int,choices=(33,35))
    witness.add_argument('seed',type=int,choices=(0,1,2))
    witness.add_argument('output',type=Path)
    args=cli.parse_args()
    if args.command=='pack':
        bundle=dict(format='beetrack-study117-v1',prereg=PREREG,coordinate_note=COORDINATE_NOTE,source_sha256={},acquisition={},
                    cases={},history={},graphs=[],mechanism={})
        add_e2e(bundle,args.root,args.score_sha,args.history)
        add_baselines(bundle,args.root,args.baseline_sha)
        add_mechanism(bundle,args.root,args.mechanism_sha)
        bundle['validation'] = {}
        for name in ('cost117','native116','physical_agreement','native_frame_agreement','retrack_index','baselines_official_check'):
            path=args.root/'validation'/(name+'.json')
            if path.is_file():
                raw=path.read_bytes()
                bundle['validation'][name]=dict(file_sha256=hashlib.sha256(raw).hexdigest(),data=json.loads(raw))
        bundle['report_sha256']=object_sha(replay(bundle))
        print(save(args.output,numerical(bundle)),args.output)
    elif args.command=='evidence':
        bundle=numerical(read(args.archive),decode=True)
        record=bundle['mechanism'][f'{args.fold}_{args.seed}']
        data=unpack_fragment(record['proof'])
        require(object_sha(data)==record['evidence_sha256'],'Incorrect case proof')
        verify_donor_rows(data)
        args.output.write_bytes(wire(data))
    else:
        output=replay(numerical(read(args.archive),decode=True))
        if args.report:
            args.report.write_bytes(wire(output))
        if args.table:
            print_table(output)
        print('Verified 36 registered P0 arms, donor coverage, comparison trackers and all source gates.')


if __name__=='__main__':
    main()
