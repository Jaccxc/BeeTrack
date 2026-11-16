"""Score all frozen E3 real-ID outputs for one role against the historical official benchmark."""
import argparse
from collections import defaultdict
import gzip
import io
import json
import os
from pathlib import Path
import re
from statistics import mean, median
from tempfile import TemporaryDirectory

import numpy as np

from beetrack.common import ROOT, RUNS
from beetrack.evaluate import load_tracks
from evaluate import metrics, score_frozen, window_metrics
from freeze_e3 import group_key, inventory, read_checked, sha, specs, wire
from inputs import LINEAGE, read_group
from tools.study103.swap_score import counts_from_events
from tools.study104.event_overlap import replay_arm


ARCHIVES = {'source':'2a7d8bf2d4b027dafa59d924e5b908ff2b88c0edbdf9fcded097c60383edd383',
            'target':'301993f5112951984cc747fb6a0466dc59bf92170ec383cbe9db7387dea9049c'}
THRESHOLDS = {'0.5':'edges05','0.3':'edges03'}


def save(path, record):
    raw = wire(record)
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        partial=Path(tmp)/'report.json'
        partial.write_bytes(raw)
        try:
            os.link(partial,path)
        except FileExistsError:
            read_checked(path,sha(raw))
    return sha(raw)


def event_histogram(rows, nframes):
    times = np.asarray([[abs(r[1]),1 if r[1]>0 else -1] for r in rows],dtype=int).reshape(-1,2)
    return counts_from_events(times,nframes)


def check_history(archive, meta, result, gt_sha):
    group_hist = []
    for seed in meta['seeds']:
        key = group_key(meta['fold'],meta['part'],seed)
        label, raw = archive['labels'][key],archive['masks']['predictions'][key]
        native = result['native'][str(seed)]
        if (raw['mot_sha256'] != meta['mot_sha256'][str(seed)] or label['gt_mot_sha256']!=gt_sha
                or label['predictions_sha256']!=meta['prediction_sha256'][str(seed)]
                or label['gt'] != result['pool_graphs']['0.5']['gt'] or label['gt'] != result['pool_graphs']['0.3']['gt']):
            raise ValueError(f'§108 prediction provenance or GT signed events differ: {key}')
        for tau, edges_key in THRESHOLDS.items():
            graph = dict(gt=label['gt'],predictions=raw['predictions'],edges=label[edges_key])
            if metrics(replay_arm(graph),witness=True) != native[tau]['standalone']:
                raise ValueError(f'Native standalone event metrics differ from §108 at {tau}: {key}')
        pred, truth = (event_histogram(records,meta['nframes']) for records in (raw['predictions'],label['gt']))
        group_hist.append(pred)
        checked = window_metrics(pred,truth)
        if checked != native['count']:
            raise ValueError(f'Native counts or window error differ from §108: {key}')
    historical_e3 = window_metrics(sum(group_hist)/3,truth)
    if historical_e3 != result['e3']:
        raise ValueError('Fractional E3 differs from three archived §108 event streams')


def paired_summary(result):
    def avg(values):
        return mean(values) if all(v is not None for v in values) else None

    native=list(result['native'].values())
    out=dict(count=dict(e3_w_percent=result['e3']['w_percent'],integer_w_percent=result['integer']['w_percent'],
                        standalone_mean_w_percent=avg([x['count']['w_percent'] for x in native]),
                        integer_minus_e3_pp=result['w_delta_vs_e3_pp'],max_abs_delta_pp=result['w_delta_bound_pp']))
    for tau in THRESHOLDS:
        scores={}
        for name in ('f1','tp'):
            a,b=(result[x][tau][name] for x in ('canonical','geometry'))
            random=[row[tau][name] for row in result['controls']]
            base=mean(row[tau]['standalone'][name] for row in native)
            full_pool=mean(row[tau]['in_full_pool'][name] for row in native)
            lean_a,lean_b=(result['lean'][x][tau][name] for x in ('canonical','geometry'))
            ceiling=result['selector_ceiling'][tau]
            center=median(random)
            scores[name]=dict(canonical=a,geometry=b,controls_median=center,standalone_mean=base,
                              standalone_in_pool_mean=full_pool,lean_canonical=lean_a,lean_geometry=lean_b,
                              geometry_minus_canonical=b-a,geometry_minus_controls_median=b-center,
                              canonical_minus_controls_median=a-center,geometry_minus_native_mean=b-base,
                              selector_ceiling=ceiling[name],ceiling_minus_geometry=ceiling[name]-b,
                              ceiling_minus_canonical=ceiling[name]-a,
                              lean_minus_complete_geometry=lean_b-b,lean_minus_complete_canonical=lean_a-a)
        scores['changed_box_owners_mean']=mean(row[tau]['changed_box_owners'] for row in native)
        out[tau]=scores
    return out


def mean_tree(records):
    if isinstance(records[0],dict):
        return {k:mean_tree([r[k] for r in records]) for k in records[0]}
    return mean(records) if all(x is not None for x in records) else None


def score(role,outdir,prereg,index_sha,lineage_dir,runs_dir,zones,archives):
    if role not in ARCHIVES or not re.fullmatch('[0-9a-f]{40}',prereg):
        raise ValueError('Invalid role or preregistration commit')
    # inventory() can create a missing index: require the already-frozen bytes before calling it.
    index=json.loads(read_checked(outdir/f'{role}_index.json',index_sha))
    if inventory(outdir,role,prereg)!=index_sha:
        raise ValueError('Role group manifests differ from the frozen index')
    archive=json.loads(gzip.decompress(read_checked(archives/f'study108_{role}.json.gz',ARCHIVES[role])))
    if archive['masks_sha256'] != LINEAGE[role] or archive['masks']['role'] != role:
        raise ValueError('Historical package differs from the pinned prediction-only role')
    reports, paired, parts = {}, {}, defaultdict(list)
    for fold,part,base in specs(role):
        key=group_key(fold,part,base)
        pack_sha=index['groups'][key]
        pack=json.loads(read_checked(outdir/role/key/'freeze.json',pack_sha))
        streams,poly,meta=read_group(role,fold,part,base,lineage_dir,runs_dir,zones)
        if pack['meta']!=meta or meta['masks_sha256']!=archive['masks_sha256']:
            raise ValueError(f'Frozen and native input metadata differ for {key}')
        label=archive['labels'][key]
        gt_path=runs_dir/'gt'/('dev' if part=='head' else 'val')/meta['clip']/'gt/gt.txt'
        gt=load_tracks(io.BytesIO(read_checked(gt_path,label['gt_mot_sha256'])))
        result=score_frozen(streams,poly,meta,gt,pack)
        check_history(archive,meta,result,label['gt_mot_sha256'])
        paired[key]=paired_summary(result)
        report=dict(role=role,prereg=prereg,index_sha256=index_sha,group_sha256=pack_sha,group=key,
                    archive108_sha256=ARCHIVES[role],gt_sha256=label['gt_mot_sha256'],meta=meta,
                    results=result,contrasts=paired[key])
        reports[key]=save(outdir/'scores'/role/(key+'.json'),report)
        parts[f'no{fold}_hive{meta["hive"]}_{part}'].append(paired[key])
    summary=dict(role=role,prereg=prereg,index_sha256=index_sha,archive108_sha256=ARCHIVES[role],
                 group_reports_sha256=reports,groups=paired,equal_group_means={k:mean_tree(v) for k,v in parts.items()},
                 equal_group_mean_all=mean_tree(list(paired.values())))
    path=outdir/f'{role}_scores.json'
    print(f'{save(path,summary)} {path}',flush=True)
    return summary


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    stage=cli.add_subparsers(dest='stage',required=True).add_parser('score')
    stage.add_argument('role',choices=tuple(ARCHIVES))
    stage.add_argument('outdir',type=Path)
    stage.add_argument('--prereg',required=True); stage.add_argument('--index-sha',required=True)
    stage.add_argument('--lineage-dir',type=Path,default=RUNS/'study108')
    stage.add_argument('--runs-dir',type=Path,default=RUNS)
    stage.add_argument('--zones',type=Path,default=ROOT/'beetrack/zones.json')
    stage.add_argument('--archives',type=Path,default=ROOT/'docs')
    a=cli.parse_args()
    score(a.role,a.outdir,a.prereg,a.index_sha,a.lineage_dir,a.runs_dir,a.zones,a.archives)
