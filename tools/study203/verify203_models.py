"""Independent source-only 203 training/choice audit; import opens no data.

audit_models(args, helpers=None) is the explicit field entry. Caller controls
field authorization. args: root, repo; helpers may provide previous (202
verifier), replay (203 verifier), certificate (independent203 certificate).
No producer training, scorer, or selection helpers are imported by the audit.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HIVES = ('35', '33', 'W', 'G')
OPTIONS = dict(gtol=1e-10, ftol=1e-14, maxiter=2000, maxls=50)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class _JSON:
    """Bounded input buffer; omit giant arrays without constructing values."""
    def __init__(self, stream):
        self.stream, self.buffer, self.position = stream, '', 0

    def peek(self):
        if self.position == len(self.buffer):
            self.buffer, self.position = self.stream.read(65536), 0
        return self.buffer[self.position:self.position+1]

    def pop(self):
        char = self.peek()
        if char:
            self.position += 1
        return char

    def space(self):
        while self.peek() and self.peek().isspace():
            self.pop()

    def expect(self, char):
        self.space()
        assert self.pop() == char

    def string(self):
        self.expect('"')
        text, escaped = ['"'], False
        while True:
            char = self.pop()
            assert char
            text.append(char)
            if char == '"' and not escaped:
                return json.loads(''.join(text))
            escaped = char == '\\' and not escaped

    def skip(self):
        """Scan balanced value boundaries without parsing nested row proofs."""
        self.space()
        opening = self.peek()
        if opening == '"':
            self.string()
            return
        if opening in ('[', '{'):
            depth, quoted, escaped = 0, False, False
            while True:
                char = self.pop()
                assert char
                if quoted:
                    if char == '"' and not escaped:
                        quoted = False
                    escaped = char == '\\' and not escaped
                elif char == '"':
                    quoted, escaped = True, False
                elif char in '[{':
                    depth += 1
                elif char in ']}':
                    depth -= 1
                    if depth == 0:
                        return
        while self.peek() and self.peek() not in ',]} \t\r\n':
            self.pop()

    def value(self, omit=()):
        self.space()
        char = self.peek()
        if char == '"':
            return self.string()
        if char == '{':
            self.pop()
            result = {}
            self.space()
            if self.peek() == '}':
                self.pop()
                return result
            while True:
                key = self.string()
                self.expect(':')
                if key in omit:
                    self.skip()
                else:
                    result[key] = self.value(omit)
                self.space()
                delimiter = self.pop()
                if delimiter == '}':
                    return result
                assert delimiter == ','
        if char == '[':
            self.pop()
            result = []
            self.space()
            if self.peek() == ']':
                self.pop()
                return result
            while True:
                result.append(self.value(omit))
                self.space()
                delimiter = self.pop()
                if delimiter == ']':
                    return result
                assert delimiter == ','
        token = []
        while self.peek() and self.peek() not in ',]} \t\r\n':
            token.append(self.pop())
        return json.loads(''.join(token))

    def array_at(self, keys):
        if not keys:
            self.expect('[')
            self.space()
            if self.peek() == ']':
                self.pop()
                return
            while True:
                yield self.value()
                self.space()
                delimiter = self.pop()
                if delimiter == ']':
                    return
                assert delimiter == ','
        self.expect('{')
        while True:
            key = self.string()
            self.expect(':')
            if key == keys[0]:
                yield from self.array_at(keys[1:])
                return
            self.skip()
            self.space()
            delimiter = self.pop()
            assert delimiter == ',', 'missing JSON key path'

    def fields(self, wanted):
        self.expect('{')
        result = {}
        while set(result) != set(wanted):
            key = self.string()
            self.expect(':')
            if key in wanted:
                result[key] = self.value()
            else:
                self.skip()
            if set(result) == set(wanted):
                return result
            self.expect(',')


def _metadata(path, omit=('training_records',)):
    with Path(path).open(encoding='utf-8') as stream:
        return _JSON(stream).value(omit)


def _records(path, mode):
    with Path(path).open(encoding='utf-8') as stream:
        yield from _JSON(stream).array_at(['models', mode, 'training_records'])


def _fields(path, wanted):
    with Path(path).open(encoding='utf-8') as stream:
        return _JSON(stream).fields(wanted)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(root):
    """Check a manifest seal without reading any referenced payload bytes."""
    root = Path(root).resolve()
    digest = _sha(root/'manifest.json')
    assert (root/'SEALED.sha256').read_text().strip() == digest
    files = _read(root/'manifest.json')
    assert all((root/name).resolve().is_relative_to(root) for name in files)
    return digest, files


def _payload(root, files, name):
    path = Path(root)/name
    assert name in files and _sha(path) == files[name]
    return _read(path)


def _token(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _outcome(value):
    a = np.asarray(value)
    assert np.issubdtype(a.dtype, np.number) and np.isfinite(a).all()
    assert (a >= 0).all() and np.array_equal(a, np.floor(a))
    return tuple(map(int, a.ravel()))


def _stratum(value):
    s = value if isinstance(value, str) else ''.join(str(int(x)) for x in value)
    assert len(s) == 5 and set(s) <= {'0', '1'}
    return s


def _dedup(contexts):
    result, aliases, hosts, actions = [], [], set(), set()
    for context in sorted(contexts, key=lambda c: c['host_key']):
        key = context['host_key']
        assert key not in hosts
        hosts.add(key)
        paths, options = {}, []
        for option in sorted(context['options'], key=lambda o: o['action_id']):
            aid = option['action_id']
            assert aid and aid not in actions
            actions.add(aid)
            assert np.asarray(option['features10']).shape == (10,) and np.isfinite(option['features10']).all()
            assert len(_outcome(option['outcome'])) == len(_outcome(context['baseline_outcome']))
            keys = option['suffix_source_keys']
            assert keys and len({_token(k) for k in keys}) == len(keys)
            path = _token(keys)
            if path in paths:
                assert _outcome(option['outcome']) == _outcome(paths[path]['outcome'])
                aliases.append(dict(host_key=key, dropped=aid, canonical=paths[path]['action_id']))
            else:
                paths[path] = option
                options.append(option)
        result.append(dict(context, options=options))
    return result, aliases


def training_rows(contexts, supervision, identity=False):
    contexts, aliases = _dedup(contexts)
    assert set(supervision) == {c['host_key'] for c in contexts}
    rows, gaps = [], []
    for c in contexts:
        if not c['options']:
            continue
        s = supervision[c['host_key']]
        if identity:
            labels = [s['classes'][o['action_id']] for o in c['options']]
            assert set(labels) <= {'S', 'D', 'U'}
            positives = labels.count('S')
            if positives == 0 and 'U' in labels:
                continue
            kept = [o for o, label in zip(c['options'], labels) if label != 'U']
            if not kept:
                continue
            target = [float(s['classes'][o['action_id']] == 'S')/positives if positives else 0. for o in kept] + [float(positives == 0)]
            rows.append(dict(c, options=kept, kind='positive' if positives else 'known-null', loss='crossentropy', target=target))
        else:
            if not s['valid']:
                gaps.append(dict(host_key=c['host_key'], reason='invalid_owner_supervision'))
                continue
            truth = _outcome(s['target_outcome'])
            compatible = [_outcome(o['outcome']) == truth for o in c['options']] + [_outcome(c['baseline_outcome']) == truth]
            if not any(compatible):
                gaps.append(dict(host_key=c['host_key'], reason='no_compatible_choice', target_outcome=list(truth)))
                continue
            rows.append(dict(c, kind='K-correct' if compatible[-1] else 'needs-change', loss='compatible', compatible=compatible))
    return rows, dict(coverage_gaps=gaps, physical_aliases=aliases, total_hosts=len(contexts), trainable_hosts=len(rows))


def weight_rows(natural, artificial, distribution):
    counts = Counter((c['hive'], str(c['group']), c['clip'], _stratum(c['stratum'])) for c in distribution if c['options'])
    key = lambda r: (r['hive'], str(r['group']), r['clip'])
    kept = [r for r in artificial if counts[(*key(r), _stratum(r['stratum']))]]
    nb, ab = defaultdict(list), defaultdict(list)
    for r in natural:
        nb[key(r)].append(r)
    for r in kept:
        ab[key(r)].append(r)
    clips = set(nb) | set(ab)
    hives = {k[0] for k in clips}
    groups, videos = defaultdict(set), defaultdict(set)
    for h, g, v in clips:
        groups[h].add(g)
        videos[h, g].add(v)
    audit = []
    for h, g, v in sorted(clips):
        k = h, g, v
        base = 1/(len(hives)*len(groups[h])*len(videos[h, g]))
        nr, ar = nb[k], ab[k]
        share = .5 if nr and ar else 1.
        kinds = Counter(r['kind'] for r in nr)
        for r in nr:
            r.update(weight=base*share/len(kinds)/kinds[r['kind']], source_type='natural')
        strata = {_stratum(r['stratum']) for r in ar}
        mass = sum(counts[(*k, s)] for s in strata)
        kinds_by = defaultdict(Counter)
        for r in ar:
            kinds_by[_stratum(r['stratum'])][r['kind']] += 1
        for r in ar:
            s = _stratum(r['stratum'])
            r.update(weight=base*share*counts[(*k, s)]/mass/len(kinds_by[s])/kinds_by[s][r['kind']], source_type='artificial')
        audit.append(dict(hive=h, group=g, clip=v, clip_weight=base, natural_mass=sum(r['weight'] for r in nr), artificial_mass=sum(r['weight'] for r in ar)))
    distribution_by = defaultdict(Counter)
    for (h, g, v, s), n in counts.items():
        distribution_by[h, g, v][s] = n
    coverage = []
    for k, cc in sorted(distribution_by.items()):
        represented = {_stratum(r['stratum']) for r in ab[k]}
        missing = sum(n for s, n in cc.items() if s not in represented)
        coverage.append(dict(hive=k[0], group=k[1], clip=k[2], natural_hosts=sum(cc.values()), uncovered_hosts=missing, uncovered_fraction=missing/sum(cc.values()), common_strata=sorted(represented)))
    return natural+kept, dict(artificial_strata_coverage=coverage, clip_type_weights=audit, natural_rows=len(natural), artificial_rows=len(kept), natural_mass=sum(r['weight'] for r in natural), artificial_mass=sum(r['weight'] for r in kept))


def loss(theta, batches):
    value, gradient = .1*float(theta@theta), .2*theta.copy()
    for matrix, row in batches:
        z = matrix@theta
        top = float(max(z))
        exp = np.exp(z-top)
        probability = exp/exp.sum()
        lz = top+np.log(exp.sum())
        if row['loss'] == 'crossentropy':
            target = np.asarray(row['target'])
            term = lz-float(target@z)
        else:
            good = np.asarray(row['compatible'], bool)
            selected = z[good]
            maximum = max(selected)
            e = np.exp(selected-maximum)
            target = np.zeros(len(z))
            target[good] = e/e.sum()
            term = lz-maximum-np.log(e.sum())
        value += row['weight']*term
        gradient += row['weight']*(probability-target)@matrix
    return float(value), gradient


def refit(contexts, supervision, mode, artificial=(), artificial_supervision=None, heldout=None):
    assert mode in ('I', 'L', 'T')
    assert all(c['hive'] != heldout for c in [*contexts, *artificial])
    natural, coverage = training_rows(contexts, supervision, mode == 'I')
    art, acov = training_rows(artificial, artificial_supervision or {}) if mode == 'T' else ([], {})
    rows, wa = weight_rows(natural, art, contexts)
    result = dict(schema='peer-suffix203-model-v1', mode=mode, heldout=heldout, ridge=.2, status='empty', mean=None, scale=None, beta=None, intercept=None,
        source_hives=sorted({r['hive'] for r in rows}), training_records=rows, coverage=coverage, artificial_coverage=acov, weight_audit=wa)
    if not rows:
        result['optimization'] = dict(success=True, message='empty source', gradient=[], iterations=0, objective=0.)
        return result
    x, w = [], []
    for r in rows:
        for option in r['options']:
            x.append(option['features10'])
            w.append(r['weight']/len(r['options']))
    x, w = np.asarray(x), np.asarray(w)
    w /= w.sum()
    mean = w@x
    variance = w@((x-mean)**2)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.
    batches = []
    for r in rows:
        a = np.zeros((len(r['options'])+1, 11))
        a[:-1, :10] = (np.asarray([o['features10'] for o in r['options']])-mean)/scale
        a[:-1, 10] = 1.
        batches.append((a, r))
    opt = minimize(lambda theta: loss(theta, batches), np.zeros(11), method='L-BFGS-B', jac=True, options=OPTIONS)
    assert opt.success and np.isfinite(opt.x).all(), opt.message
    result.update(status='fitted', mean=mean.tolist(), scale=scale.tolist(), beta=opt.x[:10].tolist(), intercept=float(opt.x[10]),
        optimization=dict(success=True, objective=float(opt.fun), gradient=loss(opt.x, batches)[1].tolist(), iterations=int(opt.nit)))
    return result


def _same(a, b, tolerance=1e-11):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _same(a[k], b[k], tolerance)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y, tolerance)
    elif isinstance(a, float):
        assert abs(a-b) <= tolerance*max(1., abs(a), abs(b)), (a, b)
    else:
        assert a == b, (a, b)


def _project(candidates, resources, certificate):
    # Reconstruct canonical resource incidence and compare sealed source rows.
    actions = [dict(c, action_id=c['candidate_id'], suffix=[dict(s, box=s.get('box', s['row'][2:6])) for s in c['suffix']]) for c in candidates]
    names, incidence = certificate.compile_resources(actions)
    groups = defaultdict(list)
    for a, indices in zip(actions, incidence):
        for i in indices:
            groups[names[i]].append(a['action_id'])
    rebuilt = [dict(resource=list(k), candidate_ids=sorted(v), capacity=1) for k, v in sorted(groups.items(), key=lambda item: str(item[0]))]
    assert rebuilt == resources
    inverse = defaultdict(list)
    for resource in rebuilt:
        for aid in resource['candidate_ids']:
            inverse[aid].append(resource['resource'])
    result = {}
    for c in candidates:
        f = c['features10']
        assert c['G_logit'] == 2*f[0]-f[2]
        result[c['action_id']] = dict(action_id=c['action_id'], features10=f, outcome=c['outcome_windows'], suffix_source_keys=c['suffix_source_keys'], resources=inverse[c['action_id']], min_anchor_iou=f[0], first_extrapolation_error=f[2], G_logit=c['G_logit'])
    return result


def audit_models(args, helpers=None):
    helpers = {} if helpers is None else helpers
    here, root = Path(__file__).parent, Path(args.root)
    previous = helpers.get('previous') or _load('independent202_model203', Path.cwd()/'tools/study202/verify_orphans.py')
    replay = helpers.get('replay') or _load('independent203_model_replay', here/'verify203.py')
    certificate = helpers.get('certificate') or _load('independent203_model_certificate', here/'verify203_certificate.py')
    digest = previous.sealed(root/'predictions', getattr(args, 'predictions_sha', None))
    fr, frozen_files = _manifest(root/'frozen')
    ow, owner_files = _manifest(root/'owners')
    descriptors = _payload(root/'frozen', frozen_files, 'deployment.json')['groups']
    contexts, roster_by, refs = [], {}, {}
    for ref in descriptors:
        folder = root/'frozen'/ref['path']
        local_seal, local_files = _manifest(folder)
        assert local_seal == ref['seal']
        data = _payload(folder, local_files, 'catalog.json')
        options = _project(data['candidates'], data['resources'], certificate)
        seen = []
        for host in data['host_contexts']:
            if not host['option_ids']:
                continue
            slot = host['host_member_index']
            seen.extend(host['option_ids'])
            contexts.append(dict(host_key=host['host_key'], hive=data['domain'], group=data['group'], clip=data['video'], unit=data['unit'], host_member=str(data['members'][slot]['member']), host_member_index=slot, stratum=host['stratum'], baseline_outcome=host['baseline_windows'], options=[options[k] for k in host['option_ids']]))
        assert len(seen) == len(set(seen)) and set(seen) == set(options)
        roster_by[data['unit']] = dict(domain=data['domain'], members=[str(m['member']) for m in data['members']])
        refs[data['unit']] = dict(ref, local_files=local_files)
        del data, options
    contexts, all_aliases = _dedup(contexts)
    reports = []
    for heldout in HIVES:
        # Whitelist BEFORE opening owner/artificial payloads. No heldout
        # labels, artificial cases, probes, summary, capacity, or score reads.
        allowed = [h for h in HIVES if h != heldout]
        source = [c for c in contexts if c['hive'] in allowed]
        supervision, artificial, asupervision = {}, [], {}
        owner_receipts, art_receipts = {}, []
        for hive in allowed:
            path = root/'owners'/(hive+'.json')
            assert _sha(path) == owner_files[hive+'.json']
            shard = _fields(path, ('domain', 'frozen_manifest_sha256'))
            assert shard['domain'] == hive and shard['frozen_manifest_sha256'] == fr
            classes, targets = {}, {}
            for unit in replay.stream_array(path, 'units'):
                assert unit['domain'] == hive
                for member in unit['members']:
                    for a in member['actions']:
                        assert a['action_id'] not in classes
                        classes[a['action_id']] = a['labels']['0.5']['class']
                for host in unit['hosts']:
                    assert host['host_key'] not in targets
                    targets[host['host_key']] = dict(valid=host['event_target_valid'], target_outcome=host['target_windows'])
                unit = member = a = host = None
            cc = [c for c in source if c['hive'] == hive]
            assert set(targets) == {c['host_key'] for c in cc}
            assert set(classes) == {o['action_id'] for c in cc for o in c['options']}
            for c in cc:
                supervision[c['host_key']] = dict(targets[c['host_key']], classes={o['action_id']: classes[o['action_id']] for o in c['options']})
            owner_receipts[hive] = dict(file=hive+'.json', sha256=previous.sha(path), hosts=len(targets), actions=len(classes))
            del shard, classes, targets, cc
        for ref in descriptors:
            if ref['domain'] not in allowed:
                continue
            path = root/'frozen'/ref['path']/'artificial.json'
            assert _sha(path) == refs[ref['unit']]['local_files']['artificial.json']
            zero = cases = 0
            for case in replay.stream_array(path, 'cases'):
                cases += 1
                if not case['candidates']:
                    zero += 1
                    continue
                options = list(_project(case['candidates'], case['resources'], certificate).values())
                strata = {c['stratum'] for c in case['candidates']}
                assert len(strata) == 1
                key, slot = case['phase_key'], case['host_member_index']
                assert key not in asupervision
                artificial.append(dict(host_key=key, hive=ref['domain'], group=ref['group'], clip=ref['video'], unit=ref['unit'], host_member=f"{ref['unit']}|{slot}", host_member_index=slot, stratum=next(iter(strata)), baseline_outcome=case['baseline_windows'], options=options, cutoff_frame=case['cutoff_frame'], cutoff_sha256=case['cutoff_sha256'], natural_host_key=case['host_key']))
                asupervision[key] = dict(valid=True, target_outcome=case['target_windows'])
                del case, options
            art_receipts.append(dict(unit=ref['unit'], hive=ref['domain'], file=ref['path']+'/artificial.json', sha256=previous.sha(path), selected_cutoffs=cases, zero_candidate_cutoffs=zero))
        model_path = root/'predictions'/(heldout+'.model.json')
        saved = _metadata(model_path)
        assert saved['heldout'] == heldout and saved['source_domains'] == allowed
        expected_receipts = dict(owners_manifest_sha256=ow, source_owner_shards=owner_receipts, source_artificial=art_receipts, target_payload_access='no heldout owner/artificial, no summary/probe/capacity/score')
        _same(expected_receipts, saved['source_receipts'])
        own_models = {}
        for mode in ('I', 'L', 'T'):
            actual = refit(source, supervision, mode, artificial, asupervision, heldout)
            model = saved['models'][mode]
            for key in ('schema', 'mode', 'heldout', 'ridge', 'status', 'source_hives', 'coverage', 'artificial_coverage', 'weight_audit'):
                _same(actual[key], model[key])
            count = 0
            for count, record in enumerate(_records(model_path, mode), start=1):
                assert count <= len(actual['training_records'])
                _same(actual['training_records'][count-1], record)
                del record
            assert count == len(actual['training_records'])
            for key in ('mean', 'scale'):
                _same(actual[key], model[key], 1e-9)
            for key in ('beta', 'intercept'):
                _same(actual[key], model[key], 5e-5)
            _same(actual['optimization']['objective'], model['optimization']['objective'], 2e-9)
            _same(actual['optimization']['gradient'], model['optimization']['gradient'], 2e-7)
            if actual['status'] == 'fitted':
                assert model['optimization']['initialization'] == [0.]*11
                assert model['optimization']['options'] == OPTIONS and model['optimization']['success'] is True
                assert model['optimization']['solver'] == 'scipy.optimize.minimize L-BFGS-B'
                batches = []
                for r in actual['training_records']:
                    matrix = np.zeros((len(r['options'])+1, 11))
                    matrix[:-1, :10] = (np.asarray([o['features10'] for o in r['options']])-model['mean'])/model['scale']
                    matrix[:-1, 10] = 1.
                    batches.append((matrix, r))
                objective, gradient = loss(np.r_[model['beta'], model['intercept']], batches)
                _same(float(objective), model['optimization']['objective'], 2e-10)
                _same(gradient.tolist(), model['optimization']['gradient'], 2e-10)
            reports.append(dict(heldout=heldout, mode=mode, trainable_hosts=len(actual['training_records']), objective=actual['optimization']['objective'], claim='local numerical convergence only; compatible-set loss can be nonconvex'))
            own_models[mode] = {k: actual[k] for k in ('status', 'mean', 'scale', 'beta', 'intercept')}
            if actual['status'] == 'fitted':
                del batches, matrix
            del actual
        target, _ = _dedup([c for c in contexts if c['hive'] == heldout])
        masks = _read(root/'predictions'/(heldout+'.masks.json'))
        assert masks['heldout'] == heldout
        expected_roster = {(unit, member) for unit, data in roster_by.items() if data['domain'] == heldout for member in data['members']}
        roster = [(m['unit'], str(m['host_member'])) for m in masks['member_clips']]
        assert len(roster) == len(set(roster)) and set(roster) == expected_roster
        _same(target, masks['contexts'])
        scores = {arm: {} for arm in ('G', 'I', 'L', 'T')}
        for c in target:
            for option in c['options']:
                aid, f = option['action_id'], np.asarray(option['features10'])
                scores['G'][aid] = 2*f[0]-f[2]
                for arm in ('I', 'L', 'T'):
                    model = own_models[arm]
                    scores[arm][aid] = None if model['status'] == 'empty' else float(((f-model['mean'])/model['scale'])@model['beta']+model['intercept'])
        scores['M'] = replay.permute_T(target, scores['T'])
        # Decode from SEALED logits after independently checking numerical
        # refit agreement: tiny coefficient rounding must not redefine masks.
        sealed_scores = {a: {} for a in scores}
        for member in masks['member_clips']:
            for arm in scores:
                assert not set(sealed_scores[arm]) & set(member['scores'][arm])
                sealed_scores[arm].update(member['scores'][arm])
        for arm in scores:
            _same(scores[arm], sealed_scores[arm], 1e-4)
        assert replay.permute_T(target, sealed_scores['T']) == sealed_scores['M']
        permutation = []
        for c in sorted(target, key=lambda c: c['host_key']):
            source_ids = sorted(o['action_id'] for o in c['options'])
            destinations = sorted(source_ids, key=lambda aid: (hashlib.sha256(('peer-suffix-rank-null-v1|'+aid).encode()).hexdigest(), aid))
            permutation.extend(dict(host_key=c['host_key'], source=src, destination=dst, score=sealed_scores['T'][src]) for src, dst in zip(source_ids, destinations))
        assert masks['M_permutation'] == permutation
        selected = {arm: [] for arm in scores}
        for unit in sorted({c['unit'] for c in target}):
            local = [c for c in target if c['unit'] == unit]
            data = _payload(root/'frozen'/refs[unit]['path'], refs[unit]['local_files'], 'catalog.json')
            for arm in scores:
                selected[arm].extend(replay.outcome_decoder(local, sealed_scores[arm], data['candidates']))
            del data
        for arm in selected:
            selected[arm].sort()
            assert masks['outcomes'][arm]['selected'] == selected[arm]
        for member in masks['member_clips']:
            host, unit = str(member['host_member']), member['unit']
            assert member['selected']['K'] == dict(candidate_ids=[])
            eligible = {o['action_id'] for c in target if c['unit'] == unit and c['host_member'] == host for o in c['options']}
            for arm in selected:
                assert member['selected'][arm]['candidate_ids'] == [aid for aid in selected[arm] if aid in eligible]
    result = dict(passed=True, prediction_manifest=digest, models=reports, source3_only=True,
        scope='Independent source context/physical-resource projection, I CE and L/T compatible-set rows/coverage, hierarchical per-clip half/fallback five-bit stratum weights, normalization, zero11 local numerical refits, logits and six-arm masks. Parent verifies source geometry/features, artificial censor targets, owner validity and publication lineage. Nonconvex fits have no global optimum or identification claim.')
    previous.receipt(root/'verification/models.json', result)
    return result


def synthetic():
    def context(name, kind='natural', stratum='00000', clip='clip'):
        return dict(host_key=name, hive='35', group=1, clip=clip, unit='u', host_member='H', stratum=stratum, baseline_outcome=[0, 0], options=[dict(action_id=name+'/a', features10=[1., 0.]*5, outcome=[1, 0], suffix_source_keys=[[name, 0]], resources=[]), dict(action_id=name+'/b', features10=[0., 1.]*5, outcome=[0, 0], suffix_source_keys=[[name, 1]], resources=[])])
    c, a, bad = context('natural'), context('art'), context('unsupported', stratum='11111')
    supervision = lambda cc: {x['host_key']: dict(valid=True, target_outcome=[1, 0], classes={x['options'][0]['action_id']: 'S', x['options'][1]['action_id']: 'D'}) for x in cc}
    models = {mode: refit([c], supervision([c]), mode, [a, bad], supervision([a, bad]), '33') for mode in ('I', 'L', 'T')}
    assert models['T']['weight_audit']['natural_mass'] == .5 and models['T']['weight_audit']['artificial_mass'] == .5
    assert models['T']['weight_audit']['artificial_rows'] == 1
    d = context('another', clip='clip2')
    mixed = refit([c, d], supervision([c, d]), 'T', [a], supervision([a]), '33')
    assert mixed['weight_audit']['natural_mass'] == .75
    row = dict(loss='compatible', compatible=[True, False, True], weight=.7)
    matrix = np.arange(33).reshape(3, 11)/30
    theta = np.linspace(-.2, .3, 11)
    value, grad = loss(theta, [(matrix, row)])
    for j in range(11):
        e = np.eye(11)[j]*1e-6
        numeric = (loss(theta+e, [(matrix, row)])[0]-loss(theta-e, [(matrix, row)])[0])/2e-6
        assert abs(grad[j]-numeric) < 1e-8
    missing = {c['host_key']: dict(valid=True, target_outcome=[0, 1])}
    rows, coverage = training_rows([c], missing)
    assert not rows and coverage['coverage_gaps'][0]['reason'] == 'no_compatible_choice'
    unknown = supervision([c])
    unknown[c['host_key']]['classes'] = {o['action_id']: 'U' for o in c['options']}
    assert not training_rows([c], unknown, True)[0]
    # Producer comparison is explicitly SYNTHETIC ONLY, never an audit input.
    producer = _load('producer203_synthetic_contract_only', Path(__file__).with_name('fit203.py'))
    for mode in ('I', 'L', 'T'):
        saved = producer.fit([c], supervision([c]), mode, artificial=[a, bad], artificial_supervision=supervision([a, bad]), heldout='33')
        for key in ('training_records', 'coverage', 'artificial_coverage', 'weight_audit', 'mean', 'scale', 'beta', 'intercept'):
            _same(models[mode][key], saved[key], 5e-5 if key in ('beta', 'intercept') else 1e-10)
        _same(models[mode]['optimization']['objective'], saved['optimization']['objective'], 2e-9)
    return dict(passed=True, field_opened=False, finite_difference_gradient=True, I_L_T_refits=True, clip_half_fallback_strata=True, missing_and_unknown_coverage=True, synthetic_producer_contract=True)


if __name__ == '__main__':
    print(json.dumps(synthetic(), sort_keys=True))
