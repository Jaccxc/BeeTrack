"""One-unit/one-model/one-physical-state replay with bounded working memory."""
import argparse
from collections import defaultdict, Counter
from fractions import Fraction
import gzip
import importlib
import json
from pathlib import Path
import sys
import tempfile
from stream202 import Reader, verified_path
from replay202 import capacity_replay, equal, score_replay


class Archive:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        assert self.manifest['schema'] == 'orphan-continuation-sharded-v1'
        self.entries = {(e['role'], e['key']): e for e in self.manifest['shards']}
        assert len(self.entries) == len(self.manifest['shards'])

    def keys(self, role): return [key for rr, key in self.entries if rr == role]

    def path(self, role, key): return verified_path(self.root, self.entries[role, key])

    def get(self, role, key):
        with gzip.open(self.path(role, key), 'rt', encoding='utf-8') as stream: return json.load(stream)

    def contexts(self, key):
        path = self.path('model_contexts', key)
        def iterate():
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                reader = Reader(stream)
                for _ in reader.array(): yield reader.value()
                assert not reader.peek()
        return iterate


def streaming_refit(model, contexts, work_dir=None):
    """Exact registered objective, source contexts streamed every optimizer pass."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    types = ('extension',) if model['mode'] == 'L' else ('extension', 'cut')
    counts, widths = defaultdict(int), {}; row_count = 0; context_count = 0
    for context in contexts():
        row_count += len(context['options'])+1; context_count += 1
        assert context['hive'] != model['heldout']
        for row in context['options']:
            kind = row['type']; counts[kind] += 1; widths[kind] = len(row['features'])
    heads, slices, size = {}, {}, 0
    for kind in types:
        expected = model['heads'][kind]
        if not counts[kind]:
            assert expected['status'] == 'empty'; continue
        # Keep just one numeric N×10/11 matrix, not all JSON options or context matrices.
        # Matching the original NumPy reduction order also preserves constant-column behavior.
        X = np.empty((counts[kind], widths[kind])); weights = np.empty(counts[kind]); index = 0
        for context in contexts():
            for row in context['options']:
                if row['type'] == kind:
                    X[index] = row['features']; weights[index] = context['weight']/len(context['options']); index += 1
        mass = float(weights.sum()); weights /= mass
        mean = weights@X; scale = np.sqrt(weights@((X-mean)**2)); scale[scale == 0] = 1
        equal(mean.tolist(), expected['mean']); equal(scale.tolist(), expected['scale'])
        equal(mass, expected['training_mass']); del X, weights
        heads[kind] = (mean, scale); slices[kind] = slice(size, size+len(scale)+1); size += len(scale)+1
    if size:
        # Disk-backed one-fold design avoids repeated JSON decoding during optimization.
        with tempfile.TemporaryDirectory(dir=work_dir, prefix='numeric202_refit_') as temp:
            A = np.memmap(Path(temp)/'design.bin', dtype='float64', mode='w+', shape=(row_count, size)); A[:] = 0
            target = np.memmap(Path(temp)/'target.bin', dtype='float64', mode='w+', shape=(row_count,))
            starts = np.empty(context_count, dtype=np.int64); lengths = np.empty(context_count, dtype=np.int64)
            weights = np.empty(context_count); cursor = 0
            for index, context in enumerate(contexts()):
                n = len(context['options'])+1; starts[index] = cursor; lengths[index] = n; weights[index] = context['weight']
                target[cursor:cursor+n] = context['target']; assert abs(float(target[cursor:cursor+n].sum())-1) < 1e-12
                for i, row in enumerate(context['options']):
                    mean, scale = heads[row['type']]
                    A[cursor+i, slices[row['type']]] = np.r_[(np.asarray(row['features'])-mean)/scale, 1]
                cursor += n
            row_weights = np.repeat(weights, lengths)
            def objective(theta):
                logits = A@theta; maxima = np.maximum.reduceat(logits, starts)
                exp = np.exp(logits-np.repeat(maxima, lengths)); sums = np.add.reduceat(exp, starts)
                value = .20/2*float(theta@theta)+float(weights@(maxima+np.log(sums)-np.add.reduceat(target*logits, starts)))
                residual = (exp/np.repeat(sums, lengths)-target)*row_weights
                gradient = .20*theta+A.T@residual
                return value, gradient
            result = minimize(objective, np.zeros(size), jac=True, method='L-BFGS-B',
                              options=dict(gtol=1e-10, ftol=1e-14, maxiter=2000, maxls=50))
            assert result.success
            for kind, sl in slices.items():
                equal(result.x[sl][:-1].tolist(), model['heads'][kind]['beta'])
                equal(float(result.x[sl][-1]), model['heads'][kind]['intercept'])
            equal(float(result.fun), model['optimization']['objective'])
            del A, target


def run(root, code_dir, repo, work_dir=None):
    sys.path[:0] = [str(repo), str(code_dir)]
    cap = importlib.import_module('cap_orphans'); fit = importlib.import_module('fit_orphans')
    scorer = importlib.import_module('score_orphans')
    from owner_numeric202 import replay as owners
    from tools.study198.run198 import combine
    archive = Archive(root)
    for entry in archive.entries.values(): verified_path(archive.root, entry)
    from stream202 import file_sha
    for name, ref in archive.manifest.get('references', {}).items():
        assert name in ('study195', 'study197', 'study198', 'study199')
        assert file_sha(repo/'tools'/name/'evidence.json.gz') == ref['sha256']
    capacity_count = 0; minimal_units = []; minimal_results = {}
    for key in archive.keys('capacity_input'):
        unit = archive.get('capacity_input', key); result = archive.get('capacity_result', key)
        data = dict(units=[unit], results={key: result}, summary=cap.summarize([unit], {key: result}))
        capacity_count += capacity_replay(data, cap)
        minimal_units.append({k: unit[k] for k in ('unit', 'hive', 'group')})
        minimal_results[key] = {f: {k: r[k] for k in ('lower_bound', 'upper_bound', 'baseline_numerator', 'denominator')} for f, r in result.items()}
        del unit, result, data
    bounds_summary = cap.summarize(minimal_units, minimal_results)
    for key in ('heads', 'groups'): equal(bounds_summary[key], archive.get('capacity_summary', key))
    case_sets = {k: set() for k in ('confirmed', 'same', 'restored')}; owner_summary = {}; owner_classes = {}; class_hives = {}
    for key in archive.keys('owner_member'):
        unit = archive.get('owner_member', key); report = owners({'units': [unit]})
        h = unit['domain']; total = owner_summary.setdefault(h, {k: 0 for k in ('S', 'nonzero_S', 'censor', 'same', 'restored')})
        for k in total: total[k] += report[h][k]
        for member in unit['members']:
            labels = {}
            for e in member['edges']:
                labels[e['edge_id']] = e['labels']['0.5']['class']; class_hives[e['edge_id']] = h
            for cid, label in member.get('cut_labels', {}).items():
                labels['cut:'+cid] = label['0.5']; class_hives['cut:'+cid] = h
            owner_classes[unit['unit'], str(member['member'])] = labels
            for record in member['census']:
                identity = tuple(record['gt_segment_key']); case_sets['confirmed'].add(identity)
                if record['has_same_GT_continuation']: case_sets['same'].add(identity)
                if record['restores_required_endpoint_side']: case_sets['restored'].add(identity)
        del unit
    model_count = 0
    for hive in archive.keys('model_fold'):
        meta = archive.get('model_fold', hive)
        assert set(meta['source_domains']) == set(fit.DOMAINS)-{hive}
        models = {}
        for mode in ('L', 'J'):
            models[mode] = archive.get('model_head', hive+'|'+mode)
            all_labels = {k: v for labels in owner_classes.values() for k, v in labels.items()
                          if class_hives[k] != hive and (mode == 'J' or not k.startswith('cut:'))}
            if owner_classes: equal(models[mode]['source_label_classes'], all_labels)
            streaming_refit(models[mode], archive.contexts(hive+'|'+mode), work_dir); model_count += 1
        for key in archive.keys('mask_member'):
            member = archive.get('mask_member', key)
            if member['hive'] != hive: continue
            result = fit.deploy(member['options'], models['L'], models['J'], [member])
            equal(result['member_clips'][0], member)
        del models
    physical_count = 0
    for key in archive.keys('physical_metric'):
        metric = archive.get('physical_metric', key); receipt = archive.get('physical_receipt', key)
        # score_replay's graph/window portion can run without any aggregate clips.
        from replay202 import signed_bins, matching
        e = metric['evidence']; nf = receipt['identity']['nf']
        equal(signed_bins(e['gt'], nf), metric['gt_bins']); equal(signed_bins(e['predictions'], nf), metric['pred_bins'])
        for threshold, name in [('0.5', 'edges05'), ('0.3', 'edges03')]:
            tp = matching(e['predictions'], e['gt'], e[name]); item = metric['iou'][threshold]
            equal(tp, item['tp']); equal(item['pred'], metric['nP']); equal(item['gt'], metric['nG'])
            equal(item['f1'], 2*tp/max(item['pred']+item['gt'], 1))
        physical_count += 1; del metric, receipt
    # Aggregate only tiny group outputs, never all native-bin matrices or graphs.
    grouped = defaultdict(list); native_requests = {}; action_stats = {}
    for key in archive.keys('native_clip'):
        item = archive.get('native_clip', key); clip = item['clip']
        grouped[item['arm'], clip['domain'], str(clip['group'])].append(key)
        stats = action_stats.setdefault((clip['domain'], item['arm']), {k: Counter() for k in ('cuts','extensions','cut_denominators','extension_denominators')})
        for member in clip['members']:
            native_requests[clip['unit'], str(member['member']), item['arm']] = {
                k: member[k] for k in ('physical_key', 'seal', 'state') if k in member}
            labels = owner_classes.get((clip['unit'], str(member['member'])))
            if labels is not None:
                for aid, label in labels.items(): stats['cut_denominators' if aid.startswith('cut:') else 'extension_denominators'][label] += 1
                for cid in member['state']['cut_ids']: stats['cuts'][labels['cut:'+cid]] += 1
                for eid in member['state']['edge_ids']: stats['extensions'][labels[eid]] += 1
    methods = defaultdict(dict)
    for (arm, hive, group), keys in grouped.items():
        clips = []
        for key in keys:
            item = archive.get('native_clip', key); clip = item['clip']
            for member in clip['members']:
                metric = archive.get('physical_metric', member['physical_key'])
                equal(member['metrics'], scorer.compact_metric(metric)); del metric
            clips.append(clip)
        result = combine(clips); result.pop('clips')
        methods[hive].setdefault(arm, []).append(result); del clips
    expected = archive.get('score_summary', 'domains')
    for (hive, arm), stats in action_stats.items():
        if 'action_stats' in expected[hive]['methods'][arm]:
            for key, value in stats.items(): equal(dict(value), expected[hive]['methods'][arm]['action_stats'][key])
    means = defaultdict(dict)
    for hive, arms in methods.items():
        for arm, groups in arms.items():
            e3 = float(sum((Fraction(100*g['numerator'], g['denominator']) for g in groups), Fraction())/len(groups))
            f1 = {t: float(sum((sum((Fraction(2*m['tp'], max(m['pred']+m['gt'], 1)) for m in g['iou'][t]['members']), Fraction())/3 for g in groups), Fraction())/len(groups)) for t in ('0.5', '0.3')}
            equal(e3, expected[hive]['methods'][arm]['E3']); equal(f1, expected[hive]['methods'][arm]['F1'])
            means[hive][arm] = dict(E3=e3, F1=f1)
    if set(means) == {'35', '33', 'W', 'G'}:
        checks = {}
        for arm in ('L', 'J'):
            delta = {h: dict(E3=means[h][arm]['E3']-means[h]['K']['E3'],
                             F1=means[h][arm]['F1']['0.5']-means[h]['K']['F1']['0.5']) for h in means}
            checks[arm] = dict(head35_gain=delta['35']['E3'] < -1.4, head33_gain=delta['33']['E3'] < -1.3,
                               W_no_harm=delta['W']['E3'] <= 1.4,
                               F1_no_harm=all(delta[h]['F1'] >= -.005 for h in ('35', '33', 'W')))
        equal(checks, archive.get('score_summary', 'investment_checks'))
        equal({a: all(v.values()) for a, v in checks.items()}, archive.get('score_summary', 'investment_pass'))
        equal(all(means[h]['L']['E3'] < means[h][other]['E3'] for h in ('35', '33') for other in ('G', 'M')),
              archive.get('score_summary', 'L_identity_ranking_pass'))
        equal(all(means[h]['J']['E3'] < means[h][other]['E3'] for h in ('35', '33') for other in ('L', 'G')),
              archive.get('score_summary', 'J_strictly_beats_L_and_G_both_heads'))
    exact = 0
    for key in archive.keys('three_exact'):
        r = archive.get('three_exact', key)
        assert all(r[k] for k in ('exact_K', 'exact_historical_s1_IDs', 'exact_native_full_stage1')); exact += 1
    requests = archive.keys('physical_request')
    if requests:
        seen, fresh, resumed, shared = set(), 0, 0, 0
        for key in sorted(requests, key=int):
            request = archive.get('physical_request', key)
            member = native_requests[request['unit'], str(request['member']), request['arm']]
            equal(request['path'], member['physical_key']); equal(request['seal'], member['seal'])
            equal(request['cut_ids'], member['state']['cut_ids']); equal(request['edge_ids'], member['state']['edge_ids'])
            token = request['path']; equal(request['within_job_shared'], token in seen)
            if token in seen: shared += 1
            elif request['resumed_from_disk']: resumed += 1
            else: fresh += 1
            seen.add(token)
        scope = archive.get('score_summary', 'physical_scope')
        equal(scope, dict(requested_arm_members=len(requests), unique_streams=len(seen), fresh_unique_physical=fresh,
                          resumed_unique_physical=resumed, same_job_shared=shared, historical_reused=0))
        assert len(seen) == physical_count
    if ('independent', 'independent') in archive.entries:
        receipt = archive.get('independent', 'independent'); assert receipt['passed']
        for stage, seal in receipt['input_seals'].items(): equal(seal, archive.manifest['sources'][stage+'/manifest.json'])
        for phase, item in receipt['phases'].items():
            equal(item['sha256'], archive.manifest['sources']['verification/'+phase+'.json'])
            assert archive.get('independent', phase)['passed']
    return dict(capacity_certificates=capacity_count, refitted_models=model_count, physical_graphs=physical_count,
                three_exact_receipts=exact, owner_summary=owner_summary, distinct_cases={k: len(v) for k, v in case_sets.items()},
                scope='bounded numeric replay; native full-frame geometry is separate independent audit')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path); p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--code-dir', type=Path, default=Path(__file__).parent)
    p.add_argument('--work-dir', type=Path, help='temporary disk-backed one-fold design; default system temp')
    args = p.parse_args()
    print(json.dumps(run(args.archive, args.code_dir, args.repo, args.work_dir), indent=2, ensure_ascii=False))
