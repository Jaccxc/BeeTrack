"""Score one frozen E3 detector bank against repaired GT on the original clip clock."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import linear_sum_assignment

from beetrack.evaluate import by_track, f1, frame_match
from tools.study103.swap_score import counts_from_events, event_times
from tools.study104.event_overlap import archive_arm, replay_arm, score_arm
from core import export_mot, write_export


def metrics(arm, witness=False):
    tp, ng, npred = len(arm.matches), len(arm.gt), len(arm.events)
    result = dict(tp=tp, fp=npred-tp, fn=ng-tp, f1=f1(tp,npred,ng))
    if witness:
        result['matching'] = [[int(p),int(g)] for p,g in sorted(arm.matches.items())]
        result['by_direction'] = {}
        for d in ('in','out'):
            gt = sum(direction == d for direction in arm.gt.values())
            pred = sum(info[0] == d for info in arm.events.values())
            hits = sum(arm.gt[g] == d for g in arm.matches.values())
            result['by_direction'][d] = dict(tp=hits,fp=pred-hits,fn=gt-hits)
    return result


def project(graph, ids):
    # Filtering the edges of one previously chosen maximum would lose valid reassignment opportunities.
    sub = dict(gt=graph['gt'], predictions=[r for r in graph['predictions'] if r[0] in ids],
               edges=[e for e in graph['edges'] if e[0] in ids])
    return replay_arm(sub)


def selector_ceiling(graph,allowed,canonical,layout):
    """Exact TP ceiling at fixed pooled ownership and one eligible selection per disjoint block.

    A matched block needs only its one witness member; no other block can use that member.
    Conversely any ordinary matched event determines an edge from its own block to its GT.
    """
    gt_column = {tid:j for j,(tid,_) in enumerate(graph['gt'])}
    neighbors = {}
    for pred,tid in graph['edges']:
        neighbors.setdefault(pred,[]).append(tid)
    edges = np.zeros((len(allowed),len(gt_column)),dtype=bool)
    witness = {}
    chosen = [tuple(key) for key in canonical]
    if len(allowed) != len(chosen) or sum(map(len,allowed)) != len(set().union(*allowed)):
        raise ValueError('Exact selector ceiling requires disjoint eligible sets')
    for i,pool in enumerate(allowed):
        # Prefer the canonical event among equally valid witnesses; fill unused blocks with it too.
        if chosen[i] not in pool:
            raise ValueError('Canonical event is outside the block it must fill')
        for member in [chosen[i],*sorted(pool-{chosen[i]})]:
            for tid in neighbors.get(layout[member][0],()):
                col = gt_column[tid]
                edges[i,col] = True
                witness.setdefault((i,col),member)
    rows,cols = linear_sum_assignment(edges,maximize=True)
    matched = []
    for i,j in zip(rows,cols):
        if edges[i,j]:
            chosen[i] = witness[(i,j)]
            matched.append([int(i),int(graph['gt'][j][0])])
    scored = project(graph,{layout[key][0] for key in chosen})
    if len(scored.matches) != len(matched):
        raise AssertionError('A completed physical parent selection fails to attain the block matching bound')
    return dict(**metrics(scored,witness=True), selected=[list(key) for key in chosen],
                block_gt_matching=matched)


def window_metrics(observed, truth):
    # Every event occurs exactly once per offset, including boundary windows.
    total = truth.sum()
    count, actual = observed.sum(axis=0)/30, truth.sum(axis=0)/30
    return dict(w_percent=float(100*np.abs(observed-truth).sum()/total) if total else None,
                count_in_out=count.tolist(), bias_in_out=(count-actual).tolist())


def score_frozen(streams, polygon, meta, gt, pack):
    """Return paired, JSON-safe scores; the calling stage must verify its case and stage freeze SHA.

    streams and gt are native frame,id,xywh[,score]. No GT is used by selector or MOT export.
    MOT witnesses are verified from the same precision and row order as submitted files.
    """
    if pack['meta'] != meta or pack['format'] != 'e3-event-mot-v1':
        raise ValueError('Score inputs differ from frozen case metadata')
    n = meta['nframes']
    streams = {s:np.asarray(v,dtype=float).reshape(-1,7) for s,v in streams.items()}
    gt = np.asarray(gt,dtype=float)
    if gt.shape == (0,):
        gt = gt.reshape(0,6)
    if gt.ndim != 2 or gt.shape[1] < 6 or not np.isfinite(gt[:,:6]).all():
        raise ValueError('Expected finite repaired GT MOT rows')
    gt = by_track(gt)
    evidence, selection = pack['manifest'], pack['selectors']
    keys = {(r['seed'],r['row'][0]) for r in evidence['observations']}
    if set(streams) != set(meta['seeds']) or len(streams) != 3 or len(selection['controls']) != 99:
        raise ValueError('Expected a complete three-detector group and 99 controls')
    full, layout = export_mot(streams, polygon, keys)
    if [[s,i,*v] for (s,i),v in sorted(layout.items())] != evidence['id_blocks']:
        raise ValueError('Physical MOT parent identity map differs from the freeze')
    pools = [b for d in ('in','out') for b in evidence['blocks'][d] if b['canonical'] is not None]
    choices = {k:selection[k] for k in ('canonical','geometry')}
    choices.update({f'control_{r:02}':items for r,items in enumerate(selection['controls'])})
    for k in ('canonical','geometry'):
        if choices[k] != [b[k] for b in pools]:
            raise ValueError(f'Frozen {k} selector is inconsistent with its evidence')
    selected_ids = {}
    allowed = [{tuple(x['member']) for x in b['evidence']} for b in pools]
    for k,items in choices.items():
        ids = [tuple(item) for item in items]
        if (len(ids) != len(pools) or len(set(ids)) != len(ids)
                or any(key not in pool for key,pool in zip(ids,allowed))):
            raise ValueError('Frozen selector violated the one-per-block, same-second budget')
        selected_ids[k] = {layout[key][0] for key in ids}

    physical = {}
    with TemporaryDirectory() as folder:
        for k in ('canonical','geometry'):
            path = Path(folder)/f'{k}.txt'
            rows, _ = export_mot(streams, polygon, choices[k])
            write_export(path,rows)
            if hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest() != pack['mot_sha256'][k]:
                raise ValueError(f'Written {k} MOT differs from the frozen file')
            native = np.loadtxt(path,delimiter=',',ndmin=2)[:,:7] if path.stat().st_size else np.empty((0,7))
            physical[k] = by_track(native)
            columns = [0,2,3,4,5,6]
            if not np.array_equal(physical[k][:,columns], full[:,columns]):
                raise ValueError('Roundtrip changed a physical box, score or within-frame competition order')

    signed = {layout[(r['seed'],r['row'][0])][0]:r['row'][1] for r in evidence['observations']}
    def histogram(ids):
        events = [[abs(signed[i]),np.sign(signed[i])] for i in ids]
        return counts_from_events(np.asarray(events,dtype=int).reshape(-1,2),n)

    truth = counts_from_events(event_times(gt,polygon),n)
    seed_ids = {s:{layout[key][0] for key in keys if key[0]==s} for s in streams}
    raw_hist = {s:counts_from_events(event_times(v,polygon),n) for s,v in streams.items()}
    pooled_counts = sum(raw_hist.values())
    integer = histogram(selected_ids['canonical'])
    delta_thirds = int(np.abs(3*integer-pooled_counts).sum())
    if (delta_thirds != pack['counts_proof']['delta_l1_thirds'] or np.any(np.abs(3*integer-pooled_counts)>2)
            or any(not np.array_equal(histogram(ids),integer) for ids in selected_ids.values())
            or any(not np.array_equal(counts_from_events(event_times(v,polygon),n),integer) for v in physical.values())
            or any(not np.array_equal(histogram(seed_ids[s]),raw_hist[s]) for s in streams)):
        raise ValueError('Official written and frozen window counts disagree')
    full_e3 = window_metrics(pooled_counts/3,truth)
    integer_summary = window_metrics(integer,truth)
    cap = float(100*delta_thirds/(3*truth.sum())) if truth.sum() else None
    delta = integer_summary['w_percent']-full_e3['w_percent'] if truth.sum() else None
    if cap is not None and abs(delta) > cap+1e-9:
        raise AssertionError('GT window errors violate the prediction-only triangle bound')
    results = dict(gt_in_out=(truth.sum(axis=0)//30).astype(int).tolist(), e3=full_e3,
                   integer=integer_summary, w_delta_vs_e3_pp=delta, w_delta_bound_pp=cap,
                   native={str(s):dict(count=window_metrics(h,truth)) for s,h in raw_hist.items()},
                   canonical={},geometry={},lean={'canonical':{},'geometry':{}}, pool_graphs={},selector_ceiling={},
                   lean_graphs={'canonical':{},'geometry':{}},
                   controls=[{} for _ in selection['controls']])

    # Parent-block order is stable even for suppressed/split event segments.
    non_events = {base for key,(base,_) in layout.items() if key not in keys}
    for thr in (0.5,0.3):
        label = str(thr)
        arm = score_arm(gt,full,polygon,thr)
        graph = archive_arm(arm)
        if {p:(f if d=='in' else -f) for p,(d,f,_) in arm.events.items()} != signed:
            raise ValueError('Full pool event times or directions differ from the frozen evidence')
        results['pool_graphs'][label] = graph
        for k,ids in selected_ids.items():
            proj = project(graph,ids)
            if k.startswith('control_'):
                results['controls'][int(k[8:])][label] = metrics(proj)
                continue
            written = score_arm(gt,physical[k],polygon,thr)
            if (proj.edges != written.edges or proj.matches != written.matches or proj.events != written.events):
                raise ValueError('Restricted full graph did not reproduce official written-MOT edges and matching')
            results[k][label] = metrics(written,witness=True)
            lean = full[np.isin(full[:,1],list(ids|non_events))]
            lean_arm = score_arm(gt,lean,polygon,thr)
            if lean_arm.events != written.events:
                raise ValueError('Lean competition control changed which events were reported')
            results['lean'][k][label] = metrics(lean_arm,witness=True)
            results['lean_graphs'][k][label] = archive_arm(lean_arm)
        ceiling = selector_ceiling(graph,allowed,choices['canonical'],layout)
        ceiling['tp_headroom'] = {k:ceiling['tp']-results[k][label]['tp'] for k in ('canonical','geometry')}
        ceiling['f1_headroom'] = {k:ceiling['f1']-results[k][label]['f1'] for k in ('canonical','geometry')}
        if min(ceiling['tp_headroom'].values()) < 0:
            raise AssertionError('A deployed selection outscored its exact ceiling')
        results['selector_ceiling'][label] = ceiling
        owners = frame_match(gt,full,thr=thr) if len(gt) and len(full) else np.full(len(full),-1)
        start = 0
        for s,native in sorted(streams.items()):
            standalone = score_arm(gt,native,polygon,thr)
            proj = project(graph,seed_ids[s])
            local = by_track(native)
            solo_owners = frame_match(gt,local,thr=thr) if len(gt) and len(local) else np.full(len(local),-1)
            end = start+len(local)
            results['native'][str(s)][label] = dict(standalone=metrics(standalone,witness=True),
                                                    in_full_pool=metrics(proj,witness=True),
                                                    changed_box_owners=int(np.sum(solo_owners!=owners[start:end])))
            start = end
    return results


def synthetic_case():
    """Crossing duplicates, a geometry-only correct choice, a shared non-event and a final singleton."""
    polygon = [[0,0],[10,0],[10,22],[0,22]]
    def bee(t,identity,y,out=False):
        x = [8,11] if out else [11,8]
        return [[f,identity,xx,y,2,2,.8] for f,xx in zip((t-1,t),x)]
    streams = {0:bee(3,7,2)+bee(52,8,5), 1:bee(3,7,2)+bee(53,8,12),
               2:bee(3,7,10)+bee(54,8,5)+bee(70,9,15,True)+[[60,10,9,18,2,2,.8]]}
    gt = np.array(bee(3,99,2)+bee(52,100,5))[:,:6]
    meta = dict(role='source',fold=33,hive=35,part='head',first_seed=0,seeds=[0,1,2],
                clip='synthetic',nframes=150)
    return {s:np.array(v) for s,v in streams.items()}, polygon, meta, gt


def verify_synthetic(pack,streams,polygon,meta,gt):
    result = score_frozen(streams,polygon,meta,gt,pack)
    for arm, graphs in result['lean_graphs'].items():
        for tau, graph in graphs.items():
            assert metrics(replay_arm(graph),witness=True) == result['lean'][arm][tau]
    graph = result['pool_graphs']['0.5']
    layout = {(s,i):base for s,i,base,_ in pack['manifest']['id_blocks']}
    for i,choice in enumerate(pack['selectors']['controls']):
        # Directly rerun every control on its own complete exported MOT for this small fixture.
        rr,_ = export_mot(streams,polygon,choice)
        for thr in (.5,.3):
            expected = metrics(score_arm(gt,rr,polygon,thr))
            assert result['controls'][i][str(thr)] == expected
    assert [layout[(0,7)],99] in graph['edges'] and [layout[(1,7)],99] not in graph['edges']
    assert result['canonical']['0.5']['tp'] == 0 and result['geometry']['0.5']['tp'] == 1
    assert result['lean']['canonical']['0.5']['tp'] == 1
    assert result['native']['1']['0.5']['changed_box_owners'] > 0
    assert result['native']['1']['0.5']['standalone']['tp'] == 1
    assert result['native']['1']['0.5']['in_full_pool']['tp'] == 0
    assert result['integer']['count_in_out'] == [2,0]
    np.testing.assert_allclose(result['e3']['count_in_out'],[2,1/3],rtol=1e-14,atol=0)
    assert abs(abs(result['w_delta_vs_e3_pp']) - result['w_delta_bound_pp']) < 1e-9
    assert len(json.dumps(result,allow_nan=False)) > 0
    return result


def self_check():
    from freeze_e3 import freeze_group
    streams, polygon, meta, gt = synthetic_case()
    with TemporaryDirectory() as folder:
        out = Path(folder)/'case'
        freeze_group(streams,polygon,meta,out,'0'*40)
        pack = json.loads((out/'freeze.json').read_text(encoding='utf-8'))
        verify_synthetic(pack,streams,polygon,meta,gt)
        pack['mot_sha256']['canonical'] = '0'*64
        try:
            score_frozen(streams,polygon,meta,gt,pack)
        except ValueError as error:
            if 'Written canonical MOT' not in str(error):
                raise
        else:
            raise AssertionError('Unpinned physical output accepted')
        # No events or labels: 99 valid empty selections, no defined normalized count error.
        empty = {s:np.empty((0,7)) for s in streams}
        out = Path(folder)/'empty'
        freeze_group(empty,polygon,meta,out,'0'*40)
        pack = json.loads((out/'freeze.json').read_text(encoding='utf-8'))
        zero = score_frozen(empty,polygon,meta,[],pack)
        assert zero['integer']['w_percent'] is None and zero['w_delta_bound_pp'] is None
        assert zero['canonical']['0.5']['tp'] == zero['canonical']['0.5']['fn'] == 0
        assert zero['selector_ceiling']['0.5']['selected'] == []
    # Independent exhaustive comparison; block 0 can claim GT 101 or 102, while block 1
    # can only claim 101. The last block adds optional competition over 102 vs 103.
    gt_ids = [101,102,103]
    graph = dict(gt=[[g,i+1] for i,g in enumerate(gt_ids)],predictions=[[i,i,0.,0.,2.,2.] for i in range(1,7)],
                 edges=[[1,101],[2,102],[3,101],[4,101],[5,102],[6,103]])
    allowed = [{(0,1),(1,2)},{(0,3),(1,4)},{(0,5),(1,6)}]
    layout = {k:(k[1],1) for k in set().union(*allowed)}
    canonical = [min(v) for v in allowed]
    for prefix in range(len(graph['edges'])+1):
        subset = dict(graph,edges=graph['edges'][:prefix])
        best = max(len(project(subset,{layout[k][0] for k in p}).matches) for p in product(*allowed))
        certificate = selector_ceiling(subset,allowed,canonical,layout)
        assert certificate['tp'] == best and len(certificate['selected']) == len(allowed)
    assert best == 3 and len(project(graph,{layout[k][0] for k in canonical}).matches) == 2
    print('Verified 99 controls against physical MOT at IoU .5/.3, geometry and lean ownership effects, '
          'fractional E3, exact count bound and attainable selector ceiling, hashes and empty groups. Synthetic only.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['self-check'])
    parser.parse_args()
    self_check()
