"""Standard-library replay: native event graphs, eight teams and integer formulas."""
import argparse
from fractions import Fraction
from itertools import product
import json
from pathlib import Path


def close(a, b):
    assert abs(a - b) < 1e-9, (a, b)


def counts(records, nf):
    result = []
    for offset in range(30):
        phase = [[0, 0] for _ in range((nf + 25 * offset + 749) // 750)]
        for row in records:
            frame = abs(int(row[1])); sign = row[1] > 0
            assert 1 <= frame <= nf
            phase[(frame - 1 + 25 * offset) // 750][not sign] += 1
        result.extend(phase)
    return result


def cardinality(edges):
    graph = {}
    for p, g in edges:
        graph.setdefault(p, []).append(g)
    assigned = {}
    def visit(p, seen):
        for g in graph[p]:
            if g not in seen:
                seen.add(g)
                if g not in assigned or visit(assigned[g], seen):
                    assigned[g] = p
                    return True
        return False
    return sum(visit(p, set()) for p in graph)


def member_score(member):
    e = member['evidence']; nf = e['nframes']
    gt, pred = counts(e['gt'], nf), counts(e['predictions'], nf)
    assert member['integer_bins'] == dict(gt=gt, pred=pred)
    numerator = sum(abs(a - b) for x, y in zip(gt, pred) for a, b in zip(x, y))
    denominator = sum(sum(row) for row in gt)
    assert (numerator, denominator) == (member['window']['numerator'], member['window']['denominator'])
    assert (member['nG'], member['nP']) == (len(e['gt']), len(e['predictions']))
    close(member['w30_percent'], 100 * numerator / denominator)
    for threshold, edgekey in [('0.5', 'edges05'), ('0.3', 'edges03')]:
        edges = e[edgekey]
        gs, ps = ({r[0]: r[1] > 0 for r in rows} for rows in (e['gt'], e['predictions']))
        assert len(set(map(tuple, edges))) == len(edges)
        assert all(p in ps and g in gs and ps[p] == gs[g] for p, g in edges)
        tp = cardinality(edges); score = member['iou'][threshold]
        assert (score['tp'], score['fp'], score['fn']) == (tp, len(ps) - tp, len(gs) - tp)
        close(score['f1'], 2 * tp / max(len(ps) + len(gs), 1))


def ratio_check(report, value):
    assert Fraction(report['numerator'], report['denominator']) == value
    close(report['value'], float(value))


def member_fraction(member, metric):
    if metric == 'W':
        return Fraction(100 * member['window']['numerator'], member['window']['denominator'])
    t = '0.5' if metric == 'F1_05' else '0.3'
    s = member['iou'][t]
    return Fraction(2 * s['tp'], max(2 * s['tp'] + s['fp'] + s['fn'], 1))


def check_group(report, members):
    seeds = report['seeds']
    assert len(seeds) == 3
    g = members[seeds[0], 'CS']['integer_bins']['gt']
    assert all(members[s, r]['integer_bins']['gt'] == g for s in seeds for r in ('CS', 'VT'))
    gsum = sum(sum(row) for row in g)
    team_numerators = {}; exposures = {'pure': {}, 'mixed': {}}
    for bits in product((0, 1), repeat=3):
        name = ''.join('V' if bit else 'C' for bit in bits)
        team = report['teams'][name]
        keys = [(s, 'VT' if bit else 'CS') for s, bit in zip(seeds, bits)]
        assert team['arms'] == [r for _, r in keys]
        chosen = [members[key] for key in keys]
        p = [[sum(m['integer_bins']['pred'][i][j] for m in chosen) for j in range(2)] for i in range(len(g))]
        num = sum(abs(x - 3 * y) for a, b in zip(p, g) for x, y in zip(a, b))
        assert (num, 3 * gsum) == (team['E3']['numerator'], team['E3']['denominator'])
        close(team['E3']['W_percent'], 100 * num / (3 * gsum))
        ratio_check(team['mean_member_W'], sum((member_fraction(m, 'W') for m in chosen), Fraction()) / 3)
        for threshold, metric in [('0.5', 'F1_05'), ('0.3', 'F1_03')]:
            ratio_check(team['mean_member_F1'][threshold], sum((member_fraction(m, metric) for m in chosen), Fraction()) / 3)
        grouping = 'pure' if len(set(bits)) == 1 else 'mixed'
        for s, r in keys:
            key = f'{s}_{r}'; exposures[grouping][key] = exposures[grouping].get(key, 0) + 1
        team_numerators[name] = num
    assert set(report['teams']) == set(team_numerators)
    assert exposures == report['exposure']
    assert set(exposures['pure'].values()) == {1} and set(exposures['mixed'].values()) == {3}
    for metric in ('W', 'F1_05', 'F1_03'):
        means = {}
        for grouping in ('pure', 'mixed'):
            slots = sum(exposures[grouping].values())
            means[grouping] = sum((member_fraction(members[s, r], metric) * exposures[grouping][f'{s}_{r}']
                                   for s in seeds for r in ('CS', 'VT')), Fraction()) / slots
        assert means['pure'] == means['mixed']
        ratio_check(report['exact_equal_member_means'][metric], means['pure'])
    form = report['closed_form']
    totals = dict(pure=0, mixed=0, gain=0, cap=0)
    categories = {name: dict(cells=0, gain18=0, score_free_ceiling18=0) for name in ('same_direction', 'other')}
    allzero = 0
    for i in range(len(g)):
        for j in range(2):
            c = [members[s, 'CS']['integer_bins']['pred'][i][j] for s in seeds]
            v = [members[s, 'VT']['integer_bins']['pred'][i][j] for s in seeds]
            d = [y - x for x, y in zip(c, v)]; S = sum(d)
            a = abs(sum(c) + sum(v) - 6 * g[i][j])
            b0, bi = abs(S), [abs(S - 2 * x) for x in d]
            pure18 = 3 * max(a, b0); mixed18 = sum(max(a, b) for b in bi)
            gain18 = pure18 - mixed18; cap18 = max(0, 3 * b0 - sum(bi))
            same = all(x >= 0 for x in d) or all(x <= 0 for x in d)
            assert [form['delta'][k][i][j] for k in range(3)] == d
            assert (form['a6'][i][j], form['b0_6'][i][j]) == (a, b0)
            assert [form['bi_6'][k][i][j] for k in range(3)] == bi
            assert (form['pure_error18'][i][j], form['mixed_error18'][i][j], form['gain18'][i][j], form['score_free_ceiling18'][i][j], form['same_direction'][i][j]) == (pure18, mixed18, gain18, cap18, same)
            assert gain18 <= cap18
            category = categories['same_direction' if same else 'other']
            category['cells'] += 1; category['gain18'] += gain18; category['score_free_ceiling18'] += cap18
            allzero += all(x == 0 for x in d)
            for name, value in [('pure', pure18), ('mixed', mixed18), ('gain', gain18), ('cap', cap18)]:
                totals[name] += value
    assert totals['pure'] == 3 * (team_numerators['CCC'] + team_numerators['VVV'])
    assert totals['mixed'] == sum(n for name, n in team_numerators.items() if name not in ('CCC', 'VVV'))
    denominator = 18 * gsum
    for grouping in ('pure', 'mixed'):
        assert report[grouping]['error18'] == totals[grouping]
        close(report[grouping]['W_percent'], 100 * totals[grouping] / denominator)
    assert (report['improvement']['error18'], report['improvement']['denominator']) == (totals['gain'], denominator)
    close(report['improvement']['pp'], 100 * totals['gain'] / denominator)
    assert (report['score_free_ceiling']['error18'], report['score_free_ceiling']['denominator']) == (totals['cap'], denominator)
    close(report['score_free_ceiling']['pp'], 100 * totals['cap'] / denominator)
    assert report['all_zero_delta_cells'] == allzero
    for name, values in categories.items():
        actual = report['categories'][name]
        assert all(actual[k] == v for k, v in values.items())
        close(actual['gain_pp'], 100 * values['gain18'] / denominator)
        close(actual['score_free_ceiling_pp'], 100 * values['score_free_ceiling18'] / denominator)
    assert sum(v['gain18'] for v in categories.values()) == totals['gain']


def replay(summary):
    assert summary['schema'] == 'mix191-score-v1'
    members = summary['members']
    assert set(members) == {f'h{h}_s{s}_{r}' for h in (33, 35) for s in range(3, 9) for r in ('CS', 'VT')}
    for member in members.values():
        member_score(member)
    for h in (33, 35):
        head = summary['heads'][str(h)]
        for name, seeds in [('345', (3, 4, 5)), ('678', (6, 7, 8))]:
            lookup = {(s, r): members[f'h{h}_s{s}_{r}'] for s in seeds for r in ('CS', 'VT')}
            check_group(head['groups'][name], lookup)
            for recipe, team in [('CS', 'CCC'), ('VT', 'VVV')]:
                assert abs(head['groups'][name]['teams'][team]['E3']['W_percent'] - summary['published_reference'][str(h)][name][recipe]) <= .0051
        groups = list(head['groups'].values()); mean = head['mean']
        for label in ('pure', 'mixed'):
            close(mean[label], sum(g[label]['W_percent'] for g in groups) / 2)
        close(mean['improvement_pp'], sum(g['improvement']['pp'] for g in groups) / 2)
        close(mean['score_free_ceiling_pp'], sum(g['score_free_ceiling']['pp'] for g in groups) / 2)
        for recipe, team in [('CS', 'CCC'), ('VT', 'VVV')]:
            close(mean[recipe], sum(g['teams'][team]['E3']['W_percent'] for g in groups) / 2)
            assert abs(mean[recipe] - summary['published_reference'][str(h)]['mean'][recipe]) <= .0051
        for name in ('same_direction', 'other'):
            for metric in ('gain_pp', 'score_free_ceiling_pp'):
                close(mean['categories'][name][metric], sum(g['categories'][name][metric] for g in groups) / 2)
        close(mean['improvement_pp'], sum(x['gain_pp'] for x in mean['categories'].values()))
    return len(members)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    package = json.loads(args.evidence.read_text(encoding='utf-8'))
    summary = package.get('summary', package)
    if 'frozen_member_events' in package:
        assert set(package['frozen_member_events']) == set(summary['members'])
        for key, records in package['frozen_member_events'].items():
            member = summary['members'][key]
            assert [[e['frame'], e['direction']] for e in records] == member['pred_events']
            assert [[e['track_id'], e['frame'] * e['direction']] for e in records] == [
                row[:2] for row in member['evidence']['predictions']]
    print(f'Replayed {replay(summary)} native members, eight-team groups, exact exposure, closed forms and score-free ceilings')
