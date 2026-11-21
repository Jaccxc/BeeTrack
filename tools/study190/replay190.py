"""Standard-library numeric replay of §190 scores and endpoint interval bounds."""
import argparse
import json
from pathlib import Path


def close(a, b):
    assert abs(a - b) < 1e-9, (a, b)


def bins(events, nf=3950):
    result = []
    for offset in range(30):
        counts = [[0, 0] for _ in range((nf + 25 * offset + 749) // 750)]
        for frame, sign in events:
            assert 1 <= frame <= nf and sign in (-1, 1)
            counts[(frame - 1 + 25 * offset) // 750][sign < 0] += 1
        result.extend(counts)
    return result


def signed_events(records):
    return [(abs(int(row[1])), 1 if row[1] > 0 else -1) for row in records]


def add(arrays):
    arrays = list(arrays)
    assert arrays and len({len(a) for a in arrays}) == 1
    return [[sum(a[i][j] for a in arrays) for j in range(2)] for i in range(len(arrays[0]))]


def matching(edges):
    neighbors = {}
    for p, g in edges:
        neighbors.setdefault(p, []).append(g)
    owner = {}

    def augment(p, seen):
        for g in neighbors[p]:
            if g not in seen:
                seen.add(g)
                if g not in owner or augment(owner[g], seen):
                    owner[g] = p
                    return True
        return False

    return sum(augment(p, set()) for p in neighbors)


def score(value):
    evidence = value['evidence']
    truth, prediction = evidence['gt'], evidence['predictions']
    gt = bins(signed_events(truth), evidence['nframes'])
    pred = bins(signed_events(prediction), evidence['nframes'])
    assert value['integer_bins'] == dict(gt=gt, pred=pred)
    denominator = 30 * len(truth)
    numerator = sum(abs(a - b) for x, y in zip(gt, pred) for a, b in zip(x, y))
    assert numerator == value['window']['numerator'] and denominator == value['window']['denominator']
    close(value['w30_percent'], 100 * numerator / denominator)
    assert (value['nG'], value['nP']) == (len(truth), len(prediction))
    gsign, psign = ({r[0]: r[1] > 0 for r in stream} for stream in (truth, prediction))
    for threshold, edgekey in [('0.5', 'edges05'), ('0.3', 'edges03')]:
        edges = evidence[edgekey]
        assert len(set(map(tuple, edges))) == len(edges)
        assert all(p in psign and g in gsign and psign[p] == gsign[g] for p, g in edges)
        tp = matching(edges)
        actual = value['iou'][threshold]
        assert (actual['tp'], actual['fp'], actual['fn']) == (tp, len(prediction) - tp, len(truth) - tp)
        close(actual['f1'], 2 * tp / max(len(prediction) + len(truth), 1))


def e3(values):
    assert len(values) == 3
    g = values[0]['integer_bins']['gt']
    assert all(v['integer_bins']['gt'] == g for v in values)
    p = add(v['integer_bins']['pred'] for v in values)
    numerator = sum(abs(x - 3 * y) for a, b in zip(p, g) for x, y in zip(a, b))
    denominator = 3 * sum(sum(row) for row in g)
    return 100 * numerator / denominator


def check_bounds(cell):
    nf = cell['scores']['base']['evidence']['nframes']
    g = cell['scores']['base']['integer_bins']['gt']
    for family in cell['bounds'].values():
        lower = [[0, 0] for _ in g]
        upper = [[0, 0] for _ in g]
        for choices in family['menus'].values():
            counts = [bins([] if event is None else [event], nf) for event in choices]
            for i in range(len(g)):
                for j in range(2):
                    lower[i][j] += min(c[i][j] for c in counts)
                    upper[i][j] += max(c[i][j] for c in counts)
        assert lower == family['lower'] and upper == family['upper']
        numerator = sum(max(lo - target, target - hi, 0) for a, b, c in zip(lower, upper, g)
                        for lo, hi, target in zip(a, b, c))
        denominator = sum(sum(row) for row in g)
        assert (numerator, denominator) == (family['bound']['numerator'], family['bound']['denominator'])
        close(family['bound']['W_lower_percent'], 100 * numerator / denominator)
        close(family['maximum_single_gain'], cell['scores']['base']['w30_percent'] - 100 * numerator / denominator)
        for arm in ('base', 'witness'):
            assert all(lo <= value <= hi for a, b, c in zip(lower, upper, cell['scores'][arm]['integer_bins']['pred'])
                       for lo, hi, value in zip(a, b, c))


def replay(package):
    summary, cells = package['summary'], package['cells']
    assert set(cells) == {f'h{h}_s{s}' for h in (33, 35) for s in range(3, 9)}
    for cell in cells.values():
        for arm in ('base', 'witness'):
            score(cell['scores'][arm])
        check_bounds(cell)
    for hive in (33, 35):
        lookup = lambda seed, arm: cells[f'h{hive}_s{seed}']['scores'][arm]
        agg = summary['aggregation'][str(hive)]
        for arm in ('base', 'witness'):
            close(agg[arm]['mean_single_W'], sum(lookup(s, arm)['w30_percent'] for s in range(3, 9)) / 6)
            for threshold, name in [('0.5', 'mean_F1_05'), ('0.3', 'mean_F1_03')]:
                close(agg[arm][name], sum(lookup(s, arm)['iou'][threshold]['f1'] for s in range(3, 9)) / 6)
            group_values = []
            for name, seeds in [('345', (3, 4, 5)), ('678', (6, 7, 8))]:
                value = e3([lookup(s, arm) for s in seeds])
                close(agg[arm]['E3'][name]['W_percent'], value)
                group_values.append(value)
            close(agg[arm]['mean_E3'], sum(group_values) / 2)
        for family in ('universal_all_side', 'gt_evidenced'):
            report = summary['bound_maximum_gains'][str(hive)][family]
            mean_lower = sum(cells[f'h{hive}_s{s}']['bounds'][family]['bound']['W_lower_percent'] for s in range(3, 9)) / 6
            close(report['mean_single_W_lower'], mean_lower)
            close(report['maximum_mean_single_gain'], agg['base']['mean_single_W'] - mean_lower)
            e3_lowers, e3_gains = [], []
            for name, seeds in [('345', (3, 4, 5)), ('678', (6, 7, 8))]:
                lower = add(cells[f'h{hive}_s{s}']['bounds'][family]['lower'] for s in seeds)
                upper = add(cells[f'h{hive}_s{s}']['bounds'][family]['upper'] for s in seeds)
                target = add([lookup(seeds[0], 'base')['integer_bins']['gt']] * 3)
                numerator = sum(max(lo - g, g - hi, 0) for a, b, c in zip(lower, upper, target) for lo, hi, g in zip(a, b, c))
                denominator = sum(sum(row) for row in target)
                bound = report['E3'][name]
                assert (numerator, denominator) == (bound['numerator'], bound['denominator'])
                value = 100 * numerator / denominator
                close(bound['W_lower_percent'], value)
                close(bound['maximum_E3_gain'], agg['base']['E3'][name]['W_percent'] - value)
                e3_lowers.append(value); e3_gains.append(bound['maximum_E3_gain'])
            close(report['mean_E3_W_lower'], sum(e3_lowers) / 2)
            close(report['maximum_mean_E3_gain'], sum(e3_gains) / 2)
        gate = summary['gate'][str(hive)]
        identities = set()
        for seed in range(3, 9):
            labels = cells[f'h{hive}_s{seed}']['labels']
            identities.update(labels['event_changed_query_gt_ids'])
        assert sorted(identities) == gate['changed_gt_segment_ids']
        group_gain = {name: sum(lookup(s, 'base')['w30_percent'] - lookup(s, 'witness')['w30_percent'] for s in seeds) / 3
                      for name, seeds in [('345', (3, 4, 5)), ('678', (6, 7, 8))]}
        mean_gain = sum(group_gain.values()) / 2
        f1_delta = agg['witness']['mean_F1_05'] - agg['base']['mean_F1_05']
        close(mean_gain, gate['mean_single_gain']); close(f1_delta, gate['official_f1_delta'])
        checks = dict(mean_single_gain_2pp=mean_gain >= 2, official_f1_nondecrease=f1_delta >= 0,
                      both_groups_positive=all(v > 0 for v in group_gain.values()),
                      at_least_5_changed_gt_identities=len(identities) >= 5)
        assert checks == gate['checks'] and all(checks.values()) == gate['pass_gate']
    return len(cells)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--evidence', type=Path)
    source.add_argument('--score-dir', type=Path)
    args = parser.parse_args()
    if args.evidence:
        package = json.loads(args.evidence.read_text(encoding='utf-8'))
    else:
        root = args.score_dir
        summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
        cells = {f'h{h}_s{s}': {name: json.loads((root / f'h{h}_s{s}' / f'{name}.json').read_text(encoding='utf-8'))
                               for name in ('scores', 'bounds', 'labels')} for h in (33, 35) for s in range(3, 9)}
        package = dict(summary=summary, cells=cells)
    print(f'Replayed {replay(package)} cells, score graphs, endpoint intervals, E3 and gates')
