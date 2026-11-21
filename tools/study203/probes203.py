"""Pure numeric all-GT-run return denominator and annotated observer checks."""
from collections import Counter


def earliest_return(observations):
    if not observations: return None
    frames = [r['frame'] for r in observations]
    assert all(type(v) is int for v in frames) and all(a < b for a, b in zip(frames, frames[1:]))
    initial = bool(observations[0]['inside'])
    crossings = [j for j in range(1, len(observations)) if bool(observations[j]['inside']) != bool(observations[j-1]['inside'])]
    for index, j in enumerate(crossings[:-1]):
        if bool(observations[j]['inside']) == initial: continue
        r = crossings[index+1]
        if bool(observations[r]['inside']) == initial and frames[r]-frames[j] <= 25:
            return dict(departure_frame=frames[j], return_frame=frames[r], initial_inside=initial)
    return None


def replay(universe, groups):
    assert universe['schema'] == 'peer-suffix203-independent-GT-universe-v1'
    eligible, all_runs = {}, set()
    for video, clip in universe['clips'].items():
        seen_rows = set()
        for run in clip['runs']:
            key = (video, run['canonical_gt_id']); assert key not in all_runs; all_runs.add(key)
            observations = run['observations']
            for row in observations:
                assert row['source_raw_index'] not in seen_rows; seen_rows.add(row['source_raw_index'])
                assert isinstance(row['row_sha256'], str) and len(row['row_sha256']) == 64
            actual = earliest_return(observations)
            saved = run['eligible_probe_or_null']
            assert (actual is None) == (saved is None)
            if actual:
                for k, value in actual.items(): assert saved[k] == value
                eligible[key] = actual
    dist = {t: {k: Counter() for k in ('geometric', 'confirmed')} for t in ('0.5', '0.3')}
    statuses = {t: Counter() for t in dist}
    observed_groups = set(); probe_observations = 0
    domains = {}
    for group in groups:
        assert group['unit'] not in observed_groups; observed_groups.add(group['unit'])
        video = group['video']; expected = {k[1] for k in eligible if k[0] == video}
        domain = domains.setdefault(group['domain'], dict(runs=set(), observations=0, distributions={t:{k:Counter() for k in ('geometric','confirmed')} for t in dist}))
        domain['runs'].update((video,k) for k in expected)
        probes = group['probes']; assert {p['canonical_gt_id'] for p in probes} == expected
        assert len(probes) == len(expected)
        for probe in probes:
            required = eligible[video, probe['canonical_gt_id']]
            for key, value in required.items(): assert probe[key] == value
            for threshold, value in probe['thresholds'].items():
                assert {m['member_index'] for m in value['members']} == {0,1,2}
                geometric, confirmed = 0, 0
                for member in value['members']:
                    rows = member['neighbor_rows']; assert member['GT_degree'] == len(rows)
                    assert len({r['union_row'] for r in rows}) == len(rows)
                    for row in rows:
                        assert any(g['canonical_gt_id'] == probe['canonical_gt_id'] and g['iou'] >= float(threshold) for g in row['graph_gt'])
                    hit = bool(rows); unique = any(r['qualified'] and r['official_owner'] == probe['canonical_gt_id'] for r in rows)
                    assert member['has_geometric_box'] == hit and member['confirmed_same_GT'] == unique
                    if not hit: status = 'zero_hit'
                    elif unique: status = 'confirmed_same_GT'
                    elif len(rows)>1 or any(r.get('pred_degree', 1)>1 for r in rows): status = 'competing_or_nonunique'
                    else: status = 'other_unconfirmed'
                    statuses[threshold][status] += 1
                    geometric += hit; confirmed += unique
                assert value['geometric_members'] == geometric and value['confirmed_members'] == confirmed
                dist[threshold]['geometric'][geometric] += 1; dist[threshold]['confirmed'][confirmed] += 1
                domain['distributions'][threshold]['geometric'][geometric] += 1
                domain['distributions'][threshold]['confirmed'][confirmed] += 1
            probe_observations += 1
            domain['observations'] += 1
    return dict(all_GT_runs=len(all_runs), distinct_eligible_runs=len(eligible), group_probe_observations=probe_observations,
                distributions={t:{k:{str(i):c[i] for i in range(4)} for k,c in vv.items()} for t,vv in dist.items()},
                member_status_counts={t:dict(c) for t,c in statuses.items()},
                domains={h:dict(distinct_GT_runs=len(d['runs']),group_observations=d['observations'],distributions={t:{k:{str(i):c[i] for i in range(4)} for k,c in vv.items()} for t,vv in d['distributions'].items()}) for h,d in domains.items()},
                scope='dates/annotated-side denominator and annotated complete-neighbor counts; full IoU geometry remains native independent scope')
