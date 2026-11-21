"""Numeric replay only; never opens caches, pixels, GT files or native MOT."""
import argparse
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isclose(a, b, rel_tol=2e-8, abs_tol=2e-9), (a, b)
    else:
        assert a == b, (a, b)


def capacity_replay(data, cap):
    totals = 0
    for unit in data['units']:
        for family, result in data['results'][unit['unit']].items():
            problem = cap.compile_problem(unit, family)
            cap.verify_certificate(problem, result['certificate'])
            upper, prediction, x = cap.witness(problem, result['selected_action_ids'])
            lower = result['certificate']['lower_bound']
            assert lower <= upper
            equal(result['baseline_numerator'], int(abs(problem['r']).sum()))
            equal(result['lower_bound'], lower)
            equal(result['upper_bound'], upper)
            equal(result['gap'], upper-lower)
            equal(result['optimal'], lower == upper)
            equal(result['witness_pred_sum'], prediction)
            equal(result['denominator'], 3*int(problem['gt'].sum()))
            equal(result['parent_keys'], problem['parents'])
            equal(result['orphan_keys'], problem['orphans'])
            equal(result['action_order'], [a['action_id'] for a in problem['actions']])
            masks = {str(m): {'cuts': [], 'edges': []} for m in unit['member_indices']}
            for a, selected in zip(problem['actions'], x):
                if selected:
                    kind = 'cuts' if a['kind'] == 'cut' else 'edges'
                    key = a.get('cut_id', a.get('candidate_id', a['action_id'].removeprefix('cut:'))) if kind == 'cuts' else a.get('edge_id', a['action_id'])
                    masks[str(a['member_index'])][kind].append(key)
            equal(result['member_masks'], masks)
            totals += 1
    equal(data['summary'], cap.summarize(data['units'], data['results']))
    return totals


def model_replay(data, model):
    options = data['options']
    for hive, fold in data['folds'].items():
        source = [r for r in options if r['hive'] != hive]
        assert set(fold['source_domains']) == set(model.DOMAINS)-{hive}
        for mode in ('L', 'J'):
            rows = source if mode == 'J' else [r for r in source if r['type'] == 'extension']
            labels = {r['action_id']: fold[mode]['source_label_classes'][r['action_id']] for r in rows}
            fitted = model.fit(source, labels, mode, hive)
            equal(fitted['contexts'], fold[mode]['contexts'])
            equal(fitted['heads'], fold[mode]['heads'])
            equal(fitted['optimization']['objective'], fold[mode]['optimization']['objective'])
        target = [r for r in options if r['hive'] == hive]
        output = model.deploy(target, fold['L'], fold['J'], [r for r in data.get('skeleton', []) if r['hive'] == hive])
        equal(output, data['masks'][hive])
    return len(data['folds'])*2


def signed_bins(events, nf):
    assert len({e[0] for e in events}) == len(events)
    rows = []
    for offset in range(30):
        phase = [[0, 0] for _ in range((nf+25*offset+749)//750)]
        for _, date, *_ in events:
            assert type(date) is int and 1 <= abs(date) <= nf
            phase[(abs(date)-1+25*offset)//750][int(date < 0)] += 1
        rows.extend(phase)
    return rows


def matching(predictions, truth, edges):
    ps, gs = {p[0] for p in predictions}, {g[0] for g in truth}
    adjacent, owner = {p: [] for p in ps}, {}
    for p, g in edges:
        assert p in ps and g in gs
        adjacent[p].append(g)
    def visit(p, seen):
        for g in adjacent[p]:
            if g in seen: continue
            seen.add(g)
            if g not in owner or visit(owner[g], seen):
                owner[g] = p
                return True
        return False
    for p in sorted(ps): visit(p, set())
    return len(owner)


def score_replay(data, scorer):
    from tools.study198.run198 import combine
    count = 0
    for value in data['physical'].values():
        metric, nf = value['metrics'], value['nf']
        evidence = metric['evidence']
        equal(signed_bins(evidence['gt'], nf), metric['gt_bins'])
        equal(signed_bins(evidence['predictions'], nf), metric['pred_bins'])
        assert len(evidence['gt']) == metric['nG'] and len(evidence['predictions']) == metric['nP']
        for threshold, key in [('0.5', 'edges05'), ('0.3', 'edges03')]:
            item = metric['iou'][threshold]
            tp = matching(evidence['predictions'], evidence['gt'], evidence[key])
            equal(item['tp'], tp)
            equal(item['pred'], metric['nP']); equal(item['gt'], metric['nG'])
            equal(item['f1'], 2*tp/max(metric['nP']+metric['nG'], 1))
        count += 1
    for arm, clips in data['native_scores'].items():
        for clip in clips:
            for member in clip['members']:
                equal(member['metrics'], scorer.compact_metric(data['physical'][member['physical_key']]['metrics']))
    rebuilt = scorer.summarize(data['native_scores'], combine, field_layout=data.get('field_layout', True))
    for key, value in rebuilt.items(): equal(value, data['summary'][key])
    return count


def replay(payload, code_dir, repo):
    sys.path.insert(0, str(repo)); sys.path.insert(0, str(code_dir))
    components = payload['components']
    assert payload['schema'] == 'orphan-continuation-public-numeric-v1'
    cap = importlib.import_module('cap_orphans')
    model = importlib.import_module('fit_orphans')
    scorer = importlib.import_module('score_orphans')
    from owner_numeric202 import replay as replay_owners
    owner_summary = replay_owners(components['owners'])
    for record in components['three_exact_receipts']:
        assert all(record['receipt'][k] for k in ('exact_K', 'exact_historical_s1_IDs', 'exact_native_full_stage1'))
    return {'union_owner_and_censor': owner_summary, 'three_exact_receipts': len(components['three_exact_receipts']),
            'capacity_certificates': capacity_replay(components['capacity'], cap),
            'refitted_models': model_replay(components['predictions'], model),
            'physical_numeric_graphs': score_replay(components['scores'], scorer),
            'scope': 'numeric source refit, integer matching certificates, dated events and annotated threshold graphs; no native geometry rematch'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--code-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--references', type=Path, help='tools directory containing pinned study archives')
    parser.add_argument('--work-dir', type=Path, help='temporary disk-backed one-fold design; default system temp')
    args = parser.parse_args()
    if args.evidence.suffix.lower() == '.zip':
        from transport202 import replay_zip
        print(json.dumps(replay_zip(args.evidence, args.code_dir, args.repo, args.work_dir), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    if args.evidence.is_dir():
        from replay_sharded202 import run
        print(json.dumps(run(args.evidence, args.code_dir, args.repo, args.work_dir), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    packed = args.evidence.read_bytes()
    manifest = json.loads(args.evidence.with_name('manifest.json').read_text())
    assert hashlib.sha256(packed).hexdigest() == manifest['files'][args.evidence.name]
    raw = gzip.decompress(packed)
    assert hashlib.sha256(raw).hexdigest() == manifest['inflated_sha256']
    payload = json.loads(raw)
    references = args.references or args.repo/'tools'
    for name, entry in payload['references'].items():
        assert name in ('study195', 'study197', 'study198', 'study199')
        assert hashlib.sha256((references/name/'evidence.json.gz').read_bytes()).hexdigest() == entry['sha256']
    print(json.dumps(replay(payload, args.code_dir, args.repo), ensure_ascii=False, indent=2))
