"""Deduplicated numeric union-owner evidence; no geometry/file IO."""
from collections import Counter


def compact_member(member):
    tables = {'0.5': {}, '0.3': {}}
    edges = []
    for original in member['edges']:
        edge = {k: v for k, v in original.items() if k != 'labels'}
        edge['labels'] = {}
        for threshold, label in original['labels'].items():
            rows = [label['recipient_row'], *label['orphan_rows']]
            for row in rows:
                key = str(row['union_row'])
                if key in tables[threshold]: assert tables[threshold][key] == row
                tables[threshold][key] = row
            edge['labels'][threshold] = {k: v for k, v in label.items() if k not in ('recipient_row', 'orphan_rows')}
            edge['labels'][threshold].update(recipient_ref=str(rows[0]['union_row']), orphan_refs=[str(r['union_row']) for r in rows[1:]])
        edges.append(edge)
    return dict(member=member['member'], member_index=member['member_index'], rows=tables, edges=edges,
                supported_cuts=[dict(candidate_id=c['candidate_id'], censor=c['labels']['0.5']['censor'])
                                for c in member['cuts'] if c['labels']['0.5']['censor']['supported']],
                census=member['truncation_opportunities'], union_proof=member['union_proof'])


def check_row(row, threshold):
    adjacent = row['graph_gt']
    assert len({g['canonical_gt_id'] for g in adjacent}) == len(adjacent) == row['pred_degree']
    assert all(g['iou'] >= threshold and g['gt_degree'] >= 1 for g in adjacent)
    reverse = row['owner_pred_neighbors']
    owner = row['official_owner']
    if owner is None:
        assert row['gt_degree'] is None and not reverse
        isolated = False
    else:
        assert len({r['union_row'] for r in reverse}) == len(reverse) == row['gt_degree']
        assert all(r['iou'] >= threshold and r['pred_degree'] >= 1 for r in reverse)
        assert sum(r['union_row'] == row['union_row'] for r in reverse) == 1
        assert sum(g['canonical_gt_id'] == owner for g in adjacent) == 1
        isolated = row['pred_degree'] == 1 and row['gt_degree'] == 1
    assert row['isolated'] == isolated
    assert row['qualified'] == (isolated and row['original_gt_id'] is not None)
    source = row['source_key']
    assert len(source) == 2 and isinstance(source[0], str) and len(source[0]) == 64 and type(source[1]) is int and source[1] >= 0


def classify(rows):
    if not all(r['qualified'] for r in rows): return 'U'
    if len({r['official_owner'] for r in rows}) == 1: return 'S'
    return 'D' if len({r['original_gt_id'] for r in rows}) > 1 else 'U'


def replay(data):
    summary = {}
    distinct = {k: set() for k in ('confirmed', 'same', 'restored')}
    per_hive = {}
    for unit in data['units']:
        h = unit['domain']; total = summary.setdefault(h, dict(S=0, nonzero_S=0, censor=0, same=0, restored=0))
        keys = per_hive.setdefault(h, {k: set() for k in distinct})
        for member in unit['members']:
            edges = member['edges']
            for threshold, rows in member['rows'].items():
                for row in rows.values(): check_row(row, float(threshold))
                for edge in edges:
                    label = edge['labels'][threshold]
                    actual = [rows[label['recipient_ref']], *[rows[k] for k in label['orphan_refs']]]
                    assert label['class'] == classify(actual)
                    if threshold == '0.5' and label['class'] == 'S':
                        total['S'] += 1
                        total['nonzero_S'] += bool(edge['v_values']) if 'v_values' in edge else any(v for r in edge['delta_windows'] for v in r)
            census = {r['candidate_id']: r for r in member['census']}
            assert set(census) == {r['candidate_id'] for r in member['supported_cuts']}
            for cut in member['supported_cuts']:
                c = cut['censor']; run = c['gt_run']; result = census[cut['candidate_id']]
                possible = [e for e in edges if e['parent_key'] == cut['candidate_id']]
                same, side = [], []
                expected = 'after' if c['type'] == 'later_return' else 'before'
                rows = member['rows']['0.5']
                for e in possible:
                    lab = e['labels']['0.5']; left = rows[lab['recipient_ref']]
                    if lab['class'] == 'S' and left['official_owner'] == run['canonical_gt_id']:
                        same.append(e['edge_id'])
                        outer = rows[lab['orphan_refs'][-1 if e['side'] == 'after' else 0]]
                        if e['side'] == expected and outer['gt_inside'] == run['start_inside'] and outer['pred_inside'] == run['start_inside']:
                            side.append(e['edge_id'])
                assert result['same_GT_continuation_ids'] == same and result['restores_required_endpoint_side_ids'] == side
                assert result['natural_edge_count'] == len(possible)
                assert result['has_same_GT_continuation'] == bool(same) and result['restores_required_endpoint_side'] == bool(side)
                total['censor'] += 1; total['same'] += bool(same); total['restored'] += bool(side)
                key = tuple(result['gt_segment_key'])
                keys['confirmed'].add(key); distinct['confirmed'].add(key)
                for name, selected in [('same', same), ('restored', side)]:
                    if selected: keys[name].add(key); distinct[name].add(key)
    for h, keys in per_hive.items():
        summary[h]['distinct_confirmed_cases'] = {k: len(v) for k, v in keys.items()}
    summary['all_distinct_confirmed_cases'] = {k: len(v) for k, v in distinct.items()}
    return summary
