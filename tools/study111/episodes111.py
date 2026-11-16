"""Select §111 near and same-bee far windows and freeze balanced original detector row masks."""

import argparse
from collections import Counter
from hashlib import sha256

import numpy as np

from beetrack.evaluate import by_track, zone_tracks
from tools.study105.shortside_census import _groups, _inside, _match, _matrix, _xyxy


DROP = {'B0': (1,2,3), 'B1': (4,5,6), 'S0': (1,3,5), 'S1': (2,4,6)}
EMPTY = np.array([], dtype=int)


def audit_frame(gt_boxes, cached, kept, first_boxes, origins, raw_ids, polygon):
    """Output H and gap diagnostics per GT row. kept contains local raw indices; origins are global."""
    gt_side = _inside(gt_boxes, polygon)
    raw_side = _inside(cached[:,:4], polygon)
    first_side = _inside(first_boxes, polygon)
    raw_iou = _matrix(gt_boxes, cached[:,:4])
    post_nms = cached[kept].astype(np.float32)
    kept_iou = _matrix(gt_boxes,post_nms[:,:4])
    kept_side = _inside(post_nms[:,:4],polygon)
    raw_owner, kept_owner, first_owner = (_match(m) for m in (raw_iou, kept_iou, _matrix(gt_boxes,first_boxes)))
    observations = Counter(int(x) for x in origins if x >= 0)
    records = []
    for i, side in enumerate(gt_side):
        raw_ok = raw_iou[i] >= .3
        right = raw_ok & (raw_side == side)
        kept_ok = kept_iou[i] >= .3
        kept_right = kept_ok & (kept_side == side)
        high = kept_right & (post_nms[:,4] >= .25)
        low = kept_right & (post_nms[:,4] >= .05) & (post_nms[:,4] < .25)
        only = np.flatnonzero(kept_ok)
        h, weak = None, False
        if len(only) == 1:
            col = int(only[0])
            raw = int(kept[col])
            exclusive = np.count_nonzero(kept_iou[:,col] >= .3) == 1
            row_id = int(raw_ids[raw])
            if exclusive and high[col] and kept_iou[i,col] >= .5 and observations[row_id] == 1:
                h = row_id
            weak = bool(exclusive and low[col])
        t, r, k = first_owner[i], raw_owner[i], kept_owner[i]
        hit = bool(t >= 0 and first_side[t] == side)
        competition = bool((right.any() and (r < 0 or raw_side[r] != side))
                           or (kept_right.any() and (k < 0 or kept_side[k] != side)))
        if hit:
            reason = 'observed'
        elif not raw_ok.any():
            reason = 'absent_from_cache'
        elif not right.any():
            reason = 'wrong_side_only'
        elif competition:
            reason = 'assignment_competition'
        elif not kept_right.any():
            reason = 'no_correct_post_nms'
        elif high.any():
            reason = 'post_nms_high'
        else:
            reason = 'post_nms_low'
        records.append(dict(h=h, first_hit=hit, any_raw=bool(raw_ok.any()), any_raw_correct=bool(right.any()),
                            kept_correct=bool(kept_right.any()), kept_high=bool(high.any()),
                            kept_low=bool(low.any()), exclusive_weak=weak, competition=competition, reason=reason))
    return records


def rank(hive, tid, c, side):
    return sha256(f'bee111-v1|{hive}|{tid}|{c}|{side}'.encode('utf-8')).hexdigest()


def choose(candidates, paired):
    used_gt, used_rows, frames, selected, reasons = set(), set(), set(), [], Counter()
    for item in sorted(candidates, key=lambda x: (x['rank'],x['gt_id'],x['c'],x['side'])):
        if paired and 'far_frames' not in item:
            reasons['no_far'] += 1
            continue
        times = [item['opposite_frame'],*item['near_frames']]
        ids = [item['opposite_idx'],*item['near_indices']]
        if paired:
            times.extend(item['far_frames'])
            ids.extend(item['far_indices'])
        if item['gt_id'] in used_gt:
            reasons['same_gt'] += 1
        elif used_rows.intersection(ids):
            reasons['shared_raw_row'] += 1
        elif any(f in frames for f in times):
            reasons['within_25_frames'] += 1
        else:
            used_gt.add(item['gt_id'])
            used_rows.update(ids)
            frames.update(g for f in times for g in range(f-25,f+26))
            selected.append({k:v for k,v in item.items() if paired or not k.startswith('far_')})
    return selected, dict(reasons)


def masks_for(rows, field, prefix):
    result = {}
    for phase, indices in DROP.items():
        retained = sorted(row[field][i] for row in rows for i in indices)
        if len(set(retained)) != 3*len(rows):
            raise ValueError('Duplicate deletion row in a treatment arm')
        result[f'{prefix}_{phase}'] = retained
    return result


def scan(gt, nframes, hive, polygon, observations):
    geometry = zone_tracks(gt, polygon)
    near, natural, rejected = [], [], Counter()
    gt_directions = {}
    for tid, (direction,start,stop,changes) in geometry.items():
        if not direction:
            continue
        gt_directions[tid] = direction
        if len(changes) != 1:
            rejected['multiple_crossings'] += 1
            continue
        ix = changes[0][0]
        c = int(gt[ix,0])
        if gt[ix-1,0] != c-1:
            rejected['nonadjacent_crossing'] += 1
            continue
        track = {int(gt[j,0]): j for j in range(start,stop)}
        if len(track) != stop-start:
            raise ValueError(f'Duplicate GT observations for resolved ID {tid}')
        for side, anchor, step in (('origin',c-1,-1),('destination',c,1)):
            near_frames = [anchor + step*i for i in range(8)]
            opp = anchor-step
            guarded = [opp,*near_frames]
            if min(guarded) <= 1 or max(guarded) >= nframes:
                rejected['boundary'] += 1
                continue
            if any(f not in track for f in guarded):
                rejected['missing_near_gt'] += 1
                continue
            guard_obs = observations[track[opp]]
            obs = [observations[track[f]] for f in near_frames]
            key = dict(gt_id=tid, c=c, dir=direction, side=side, rank=rank(hive,tid,c,side))
            if guard_obs['h'] is None:
                rejected['opposite_not_H'] += 1
                continue
            for phase in ('B0','B1'):
                gap = DROP[phase]
                if any(obs[j]['first_hit'] if j in gap else obs[j]['h'] is None for j in range(8)):
                    continue
                details = [dict(frame=near_frames[j], **{k:v for k,v in obs[j].items() if k != 'h'}) for j in gap]
                weak = sum(r['exclusive_weak'] for r in details) >= 2 and not any(r['kept_high'] for r in details)
                natural.append(dict(key,phase=phase, gap_frames=[r['frame'] for r in details],
                                    gap_audit=details, weak=weak))
            if any(o['h'] is None for o in obs):
                rejected['near_not_H'] += 1
                continue
            row = dict(key, opposite_frame=opp, opposite_idx=guard_obs['h'], near_frames=near_frames,
                       near_indices=[o['h'] for o in obs])
            # Near occupies offsets 0..7. Offset 33 is 26 frames from its last observation.
            for j in range(8,102):
                f = anchor + step*j
                if f not in track or abs(f-c) > 100 or f <= 1 or f >= nframes:
                    break
                if j < 40:
                    continue
                far = [anchor + step*z for z in range(j-7,j+1)]
                indexes = [observations[track[z]]['h'] for z in far]
                if all(x is not None for x in indexes):
                    row.update(far_frames=far,far_indices=indexes)
                    break
            near.append(row)
    selected, weak_reps = {}, {}
    for item in sorted(natural, key=lambda x: (x['rank'],x['gt_id'],x['c'],x['side'],x['phase'])):
        selected.setdefault(item['gt_id'],item)
        if item['weak']:
            weak_reps.setdefault(item['gt_id'],item)
    rows_all, excl_all = choose(near, False)
    rows, excl_pair = choose(near, True)
    return dict(rows_all=rows_all, rows=rows, natural_rows=list(selected.values()),
                natural_weak_rows=list(weak_reps.values()), gt_directions=gt_directions,
                masks={**masks_for(rows_all,'near_indices','nearall'), **masks_for(rows,'near_indices','near'),
                       **masks_for(rows,'far_indices','far')},
                coverage=dict(gt=len(gt_directions), near_sides=len(near), near_ids=len({r['gt_id'] for r in near}),
                              paired_sides=sum('far_frames' in r for r in near),
                              paired_ids=len({r['gt_id'] for r in near if 'far_frames' in r}),
                              near_selected=len(rows_all), paired_selected=len(rows),
                              natural_patterns=len(natural), natural_ids=len(selected),
                              exclusive_weak_ids=len(weak_reps)),
                exclusions=dict(geometry=dict(rejected), nearall=excl_all, paired=excl_pair))


def build_case(gt, polygon, detections, survivor_indices, first_rows, observed_indices, hive, nframes):
    gt = by_track(gt)
    polygon = np.asarray(polygon,np.float32)
    d = np.asarray(detections)
    survivor_indices = np.asarray(survivor_indices,dtype=np.int64)
    first_rows, observed_indices = np.asarray(first_rows), np.asarray(observed_indices,dtype=np.int64)
    if (len(first_rows) != len(observed_indices) or len(np.unique(survivor_indices)) != len(survivor_indices)
            or np.any(survivor_indices < 0) or np.any(survivor_indices >= len(d))
            or not np.isin(observed_indices[observed_indices >= 0], survivor_indices).all()):
        raise ValueError('Inconsistent native first-pass provenance or survivor indices')
    known = observed_indices >= 0
    if not np.array_equal(d[observed_indices[known],0], first_rows[known,0]):
        raise ValueError('First-pass origin refers to a different frame')
    first_boxes, gt_boxes = _xyxy(first_rows), _xyxy(gt)
    raw_groups, first_groups = _groups(d[:,0]), _groups(first_rows[:,0])
    kept_groups = _groups(d[survivor_indices,0])
    observations = {}
    for f,g in _groups(gt[:,0]).items():
        raw, first = raw_groups.get(f,EMPTY), first_groups.get(f,EMPTY)
        # Map to frame-local columns in the native NMS survivor order, including for tied assignments.
        local_kept = np.searchsorted(raw,survivor_indices[kept_groups.get(f,EMPTY)])
        audit = audit_frame(gt_boxes[g],d[raw,1:6],local_kept,first_boxes[first],observed_indices[first],raw,polygon)
        observations.update(zip(g,audit))
    return scan(gt,nframes,hive,polygon,observations)


def self_check():
    poly = np.array([[0,0],[10,0],[10,20],[0,20]],np.float32)
    gt = np.array([[f,7,12 if f < 89 else 4,5,2,2] for f in range(2,178)],float)
    det = np.c_[gt[:,0],_xyxy(gt),np.full(len(gt),.8)]
    first = np.c_[gt,np.full(len(gt),.8)]
    ids = np.arange(len(det))
    def build(raw=det, native=first, origins=ids, truth=gt, survivors=ids):
        return build_case(truth,poly,raw,survivors,native,origins,33,200)
    clean = build()
    assert clean['coverage']['near_sides'] == clean['coverage']['paired_sides'] == 2
    assert clean['gt_directions'] == {7:'in'} and not clean['natural_rows']
    for row in clean['rows']:
        assert min(abs(x-y) for x in row['near_frames'] for y in row['far_frames']) == 26
        assert set(row['near_indices']).isdisjoint(row['far_indices'])
        for place in ('near','far'):
            masks = clean['masks']
            assert set(masks[place+'_B0']+masks[place+'_B1']) == set(masks[place+'_S0']+masks[place+'_S1'])
            assert set(masks[place+'_B0']+masks[place+'_B1']) == set(row[place+'_indices'][1:7])
    # Two natural B0 examples on one GT. The earlier-ranked side lacks any detection;
    # the other side has exclusive weak rows, and must count in its own subgroup.
    by_rank = sorted(('origin','destination'),key=lambda s: rank(33,7,89,s))
    gap = {'origin':[87,86,85], 'destination':[90,91,92]}
    strongless = det.copy()
    first_ok = ~np.isin(gt[:,0],gap['origin']+gap['destination'])
    strongless[np.isin(gt[:,0],gap[by_rank[1]]),5] = .15
    seen = ~np.isin(gt[:,0],gap[by_rank[0]])
    reduced = strongless[seen]
    reduced_ids = np.arange(len(reduced))
    row_map = {int(f):i for i,f in enumerate(reduced[:,0])}
    natural = build(reduced,first[first_ok],np.asarray([row_map[int(f)] for f in gt[first_ok,0]]),survivors=reduced_ids)
    assert not natural['rows_all'] and len(natural['natural_rows']) == len(natural['natural_weak_rows']) == 1
    assert not natural['natural_rows'][0]['weak'] and natural['natural_weak_rows'][0]['weak']
    assert {r['reason'] for r in natural['natural_rows'][0]['gap_audit']} == {'absent_from_cache'}
    assert {r['reason'] for r in natural['natural_weak_rows'][0]['gap_audit']} == {'post_nms_low'}
    # Far stretch with one missing annotated intermediate frame on both sides.
    interrupted = build(truth=gt[~np.isin(gt[:,0],[65,115])])
    assert len(interrupted['rows_all']) == 1 and interrupted['rows'] == []
    bounce = gt.copy()
    bounce[gt[:,0] == 100,2] = 12
    assert build(truth=bounce)['coverage']['near_sides'] == 0
    # Shared survivor touching a second GT, independent of first-pass assignment.
    boxes = np.array([[1,1,3,3],[1,1,3,3]],float)
    cache = np.array([[1,1,3,3,.8]],float)
    contested = audit_frame(boxes,cache,np.array([0]),cache[:,:4],np.array([8]),np.array([8]),poly)
    assert all(x['h'] is None for x in contested) and sum(x['first_hit'] for x in contested) == 1
    # Nearby observations can both exist, but one batch must not intervene twice nearby.
    row = clean['rows'][0]
    conflict = dict(row,gt_id=22,rank='0'*64,near_indices=[200+i for i in range(8)],opposite_idx=298)
    selected, excluded = choose([row,conflict],False)
    assert [r['gt_id'] for r in selected] == [22] and excluded == {'within_25_frames':1}
    print('Synthetic §111 geometry, H provenance, paired selection, masks and weak natural census passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['self-check'])
    parser.parse_args()
    self_check()
