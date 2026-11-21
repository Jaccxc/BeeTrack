"""Source-only conditional choices. Default CLI runs synthetic checks only.

An option is a GT-free flat record with action_id, type (extension/cut),
features (10/11), hive/group/clip/unit/member/member_index/recipient and,
for extensions, orphan_key. Labels are a separate action_id -> S/D/U map.
L conditions on orphan; J conditions on recipient. Unknown alternatives
never enter training softmaxes; deployment always uses the complete catalogue.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import scipy
from scipy.optimize import minimize, linear_sum_assignment
from scipy.special import logsumexp

DOMAINS = ('35', '33', 'W', 'G')
DIMENSIONS = dict(extension=10, cut=11)
ARMS = ('K', 'G', 'L', 'M', 'J')
RIDGE = .20
SCHEMA = 'orphan-conditional-choice-v1'


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [plain(v) for v in x]
    return x


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(plain(obj), sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf8')


def validate(options):
    ids = set()
    for r in options:
        if r['action_id'] in ids: raise ValueError('duplicate action_id')
        ids.add(r['action_id'])
        if not isinstance(r['action_id'], str) or not r['action_id']: raise ValueError('nonempty string action id required')
        if r['hive'] not in DOMAINS: raise ValueError('unknown hive')
        for k in ('group', 'clip', 'unit', 'member', 'member_index', 'recipient'):
            if k not in r: raise ValueError('missing option metadata: '+k)
        if r['type'] not in DIMENSIONS: raise ValueError('unknown option type')
        if r['type'] == 'extension' and 'orphan_key' not in r: raise ValueError('extension requires orphan_key')
        x = np.asarray(r['features'], float)
        if x.shape != (DIMENSIONS[r['type']],) or not np.isfinite(x).all(): raise ValueError('invalid option features')
    return options


def _context(r, mode):
    return (r['unit'], str(r['member']), str(r['orphan_key'] if mode == 'L' else r['recipient']))


def training_contexts(options, labels, mode, heldout=None):
    """Positive targets sum to one; a context contributes its weight ONCE."""
    if mode not in ('L', 'J'): raise ValueError('mode must L/J')
    validate(options)
    if heldout is not None and any(r['hive'] == heldout for r in options): raise ValueError('heldout source option')
    options = sorted((r for r in options if mode == 'J' or r['type'] == 'extension'), key=lambda r: r['action_id'])
    if set(labels) != {r['action_id'] for r in options}: raise ValueError('source label inventory must match model options')
    grouped = defaultdict(list)
    for r in options: grouped[_context(r, mode)].append(r)
    out = []
    for key, rows in sorted(grouped.items()):
        if len({(r['hive'], str(r['group']), r['clip']) for r in rows}) != 1: raise ValueError('context crosses clips/groups')
        classes = [labels[r['action_id']] for r in rows]
        if any(c not in ('S', 'D', 'U') for c in classes): raise ValueError('unknown owner label')
        positive = [c == ('S' if r['type'] == 'extension' else 'D') for r, c in zip(rows, classes)]
        if not any(positive) and 'U' in classes: continue
        known = [(r, p) for r, c, p in zip(rows, classes, positive) if c != 'U']
        if not known: continue
        npositive = sum(p for _, p in known)
        first = rows[0]
        target = [float(p)/npositive if npositive else 0. for _, p in known]+[float(not npositive)]
        out.append(dict(context=list(key), hive=first['hive'], group=str(first['group']), clip=first['clip'],
                        kind='positive' if npositive else 'known-null', options=[r for r, _ in known], target=target,
                        omitted_U=[r['action_id'] for r, c in zip(rows, classes) if c == 'U']))
    # Count each hierarchy once; all members of a clip share its contexts.
    hs=set(); gs=defaultdict(set); vs=defaultdict(set); ks=defaultdict(set); counts=Counter()
    for c in out:
        h,g,v,k=c['hive'],c['group'],c['clip'],c['kind']
        hs.add(h); gs[h].add(g); vs[h,g].add(v); ks[h,g,v].add(k); counts[h,g,v,k]+=1
    for c in out:
        h, g, v, k = c['hive'], c['group'], c['clip'], c['kind']
        c['weight'] = 1/(len(hs)*len(gs[h])*len(vs[h,g])*len(ks[h,g,v])*counts[h,g,v,k])
    if out and not np.isclose(sum(c['weight'] for c in out), 1, atol=1e-14): raise AssertionError('weights do not sum1')
    return out


def fit(options, labels, mode='L', heldout=None):
    contexts = training_contexts(options, labels, mode, heldout)
    types = ('extension',) if mode == 'L' else ('extension', 'cut')
    heads, slices, start = {}, {}, 0
    for kind in types:
        rows, weights = [], []
        for c in contexts:
            for r in c['options']:
                if r['type'] == kind:
                    rows.append(r['features']); weights.append(c['weight']/len(c['options']))
        if not rows:
            heads[kind] = dict(status='empty', mean=None, scale=None, beta=None, intercept=None)
            continue
        X, w = np.asarray(rows, float), np.asarray(weights)
        mass = float(w.sum()); w /= mass
        mean = w@X; scale = np.sqrt(w@((X-mean)**2)); scale[scale == 0] = 1.
        heads[kind] = dict(status='fitted', mean=mean.tolist(), scale=scale.tolist(), training_mass=mass)
        slices[kind] = slice(start, start+DIMENSIONS[kind]+1); start += DIMENSIONS[kind]+1
    matrices = []
    for c in contexts:
        a = np.zeros((len(c['options'])+1, start))
        for i, r in enumerate(c['options']):
            h = heads[r['type']]; sl = slices[r['type']]
            a[i, sl] = np.r_[(np.asarray(r['features'])-h['mean'])/h['scale'], 1.]
        matrices.append((a, np.asarray(c['target']), c['weight']))

    def objective(theta):
        value = RIDGE/2*float(theta@theta); grad = RIDGE*theta
        for a, target, w in matrices:
            logits = a@theta; logz = logsumexp(logits)
            value += w*(logz-target@logits)
            grad += w*a.T@(np.exp(logits-logz)-target)
        return value, grad

    result = None
    if start:
        result = minimize(objective, np.zeros(start), jac=True, method='L-BFGS-B',
                          options=dict(gtol=1e-10, ftol=1e-14, maxiter=2000, maxls=50))
        if not result.success or not np.isfinite(result.x).all(): raise RuntimeError('conditional fit did not converge: '+str(result.message))
        for kind, sl in slices.items():
            heads[kind].update(beta=result.x[sl][:-1].tolist(), intercept=float(result.x[sl][-1]))
    return dict(schema=SCHEMA, mode=mode, heldout=heldout, ridge=RIDGE,
                penalty='0.20/2*(all beta squared + edge intercepts squared)', heads=heads,
                source_hives=sorted({c['hive'] for c in contexts}), contexts=plain(contexts),
                source_label_classes=dict(sorted(labels.items())),
                optimization=dict(success=True, message=str(result.message) if result else 'empty source',
                                  iterations=int(result.nit) if result else 0,
                                  objective=float(result.fun) if result else 0.,
                                  gradient=objective(result.x)[1].tolist() if result else [],
                                  max_abs_gradient=float(np.max(np.abs(objective(result.x)[1]))) if result else 0.))


def predict(model, options):
    """Only numeric option features enter logits. Empty option type abstains."""
    validate(options); output = {}
    for r in options:
        h = model['heads'].get(r['type'])
        output[r['action_id']] = (None if h is None or h['status'] == 'empty' else
            float(((np.asarray(r['features'])-h['mean'])/h['scale'])@h['beta']+h['intercept']))
    return output


def matching(options, scores):
    """Sorted SciPy linear-sum assignment, raw binary64 logits, null=0.

    Each recipient row has its own private null column. Cut options have
    private dummy columns. Missing and nonpositive edges never enter. No
    quantization, tolerance, score clipping or artificial tie perturbation.
    """
    validate(options)
    if set(scores) != {r['action_id'] for r in options}: raise ValueError('score inventory differs')
    bymember = defaultdict(list)
    for r in options:
        value = scores[r['action_id']]
        if value is not None and not np.isfinite(value): raise ValueError('nonfinite logit')
        if value is not None and value > 0: bymember[r['unit'], str(r['member'])].append(r)
    selected = []
    for _, rows in sorted(bymember.items()):
        rows.sort(key=lambda r: r['action_id'])
        lefts = sorted({str(r['recipient']) for r in rows})
        def right(r):
            return ('orphan', str(r['orphan_key'])) if r['type'] == 'extension' else ('cut', r['action_id'])
        rights = sorted({right(r) for r in rows})+[('null', pid) for pid in lefts]
        li, ri = {k:i for i,k in enumerate(lefts)}, {k:i for i,k in enumerate(rights)}
        weights = np.full((len(lefts),len(rights)), -np.inf)
        owners = {}
        for pid in lefts: weights[li[pid],ri['null',pid]]=0.
        for r in rows:
            ij=li[str(r['recipient'])],ri[right(r)]; score=float(scores[r['action_id']])
            # Sorted action IDs settle duplicate parallel alternatives.
            if score>weights[ij]: weights[ij]=score; owners[ij]=r['action_id']
        ii,jj=linear_sum_assignment(weights,maximize=True)
        selected.extend(owners[i,j] for i,j in zip(ii,jj) if (i,j) in owners)
    return sorted(selected)


def shuffled_scores(options, scores):
    """Reassign ALL L scores, including negatives/abstentions, within orphan."""
    validate(options); grouped = defaultdict(list); out = {}; audit = []
    if any(r['type'] != 'extension' for r in options): raise ValueError('M only shuffles extension scores')
    for r in options: grouped[_context(r, 'L')].append(r)
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r['action_id'])
        targets = sorted(rows, key=lambda r: (hashlib.sha256(('orphan-recipient-null-v1|'+r['action_id']).encode()).hexdigest(), r['action_id']))
        for src, dst in zip(rows, targets):
            out[dst['action_id']] = scores[src['action_id']]
            audit.append(dict(source=src['action_id'], destination=dst['action_id'], score=scores[src['action_id']]))
    return out, audit


def deploy(options, L, J, skeleton=()):
    validate(options)
    ext = [r for r in options if r['type'] == 'extension']
    ls, js = predict(L, ext), predict(J, options)
    gs = {r['action_id']: 1-r['features'][4]/2-r['features'][0] for r in ext}
    ms, permutation = shuffled_scores(ext, ls)
    choices = dict(K=[], G=matching(ext, gs), L=matching(ext, ls), M=matching(ext, ms), J=matching(options, js))
    grouped = {(r['unit'], str(r['member'])): dict(unit=r['unit'], member=str(r['member']), member_index=r['member_index'], hive=r['hive'], clip=r['clip']) for r in skeleton}
    for r in options: grouped.setdefault((r['unit'], str(r['member'])), dict(unit=r['unit'], member=str(r['member']), member_index=r['member_index'], hive=r['hive'], clip=r['clip']))
    members = []
    for (unit, member), meta in sorted(grouped.items()):
        rows = sorted([r for r in options if (r['unit'], str(r['member'])) == (unit, member)], key=lambda r: r['action_id'])
        ids = {r['action_id'] for r in rows}
        selected = {arm: dict(edge_ids=[r['action_id'] for r in rows if r['type'] == 'extension' and r['action_id'] in choices[arm]],
                              cut_ids=[r['candidate_id'] for r in rows if r['type'] == 'cut' and r['action_id'] in choices[arm]]) for arm in ARMS}
        members.append(dict(meta, options=plain(rows), selected=selected,
                            scores={a: {k: v for k, v in s.items() if k in ids} for a, s in dict(G=gs,L=ls,M=ms,J=js).items()},
                            M_permutation=[p for p in permutation if p['source'] in ids]))
    return dict(member_clips=members, tie_rule='sorted recipient/orphan/private dummy/action IDs; scipy linear_sum_assignment maximize=True, raw binary64; private null0')


def train_and_seal(options, label_provider, outdir, input_receipt, skeleton=()):
    """Provider(target) may open only the OTHER THREE source shards."""
    validate(options)
    if any('labels' in r or 'class' in r for r in options): raise ValueError('deployment inventory must be unlabelled')
    dest = Path(outdir)
    if dest.exists(): raise FileExistsError('prediction destination exists; do not overwrite')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.'+dest.name+'-', dir=dest.parent))
    try:
        for target in DOMAINS:
            supplied = label_provider(target)
            if set(supplied['source_hives']) != set(DOMAINS)-{target}: raise ValueError('source whitelist differs')
            source = [r for r in options if r['hive'] != target]
            labels = supplied['labels']
            if set(labels) != {r['action_id'] for r in source}: raise ValueError('source labels incomplete or include target')
            L = fit(source, {r['action_id']: labels[r['action_id']] for r in source if r['type'] == 'extension'}, 'L', target)
            J = fit(source, labels, 'J', target)
            save(tmp/f'{target}.model.json', dict(heldout=target, source_domains=[h for h in DOMAINS if h != target],
                L=L, J=J, source_receipts=supplied.get('receipts', {})))
            target_options = [r for r in options if r['hive'] == target]
            output = deploy(target_options,L,J,[r for r in skeleton if r['hive'] == target])
            save(tmp/f'{target}.masks.json', dict(heldout=target, **output))
        receipt = dict(input_receipt, schema=SCHEMA, model_code_sha256=sha(__file__), arms=list(ARMS),
                       folds=list(DOMAINS), no_target_labels_before_common_seal=True, scipy_version=scipy.__version__, numpy_version=np.__version__, matching_algorithm='scipy.optimize.linear_sum_assignment maximize=True, sorted rows/columns, private null0')
        save(tmp/'receipt.json',receipt)
        save(tmp/'manifest.json',{p.name:sha(p) for p in tmp.iterdir()})
        digest = sha(tmp/'manifest.json'); (tmp/'SEALED.sha256').write_text(digest+'\n')
        tmp.rename(dest)
        return dict(root=str(dest),manifest_sha256=digest)
    finally:
        if tmp.exists(): shutil.rmtree(tmp)


def verify_seal(root):
    root=Path(root)
    if sha(root/'manifest.json') != (root/'SEALED.sha256').read_text().strip(): raise ValueError('prediction seal differs')
    manifest=json.loads((root/'manifest.json').read_text())
    if set(manifest) != {'receipt.json'}|{f'{h}.{kind}.json' for h in DOMAINS for kind in ('model','masks')}: raise ValueError('fourfold inventory differs')
    for name,digest in manifest.items():
        if sha(root/name) != digest: raise ValueError('prediction file changed')
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['model_code_sha256'] != sha(__file__): raise ValueError('model code changed after seal')
    return receipt


def load_predictions(root):
    receipt=verify_seal(root); folds={}; decisions={}
    for hive in DOMAINS:
        model=json.loads((Path(root)/f'{hive}.model.json').read_text())
        if model['heldout'] != hive or set(model['source_domains']) != set(DOMAINS)-{hive}: raise ValueError('source-only fold changed')
        fold=json.loads((Path(root)/f'{hive}.masks.json').read_text())
        if fold['heldout'] != hive: raise ValueError('mask heldout changed')
        folds[hive]=fold
        for m in fold['member_clips']:
            unit=decisions.setdefault(m['unit'],{})
            if m['member'] in unit: raise ValueError('duplicate member')
            unit[m['member']]=m['selected']
    return dict(receipt=receipt,folds=folds,decisions=decisions)


def manifest(root, verify_files=False):
    root=Path(root); digest=sha(root/'manifest.json')
    if (root/'SEALED.sha256').read_text().strip()!=digest: raise ValueError('input seal differs')
    files=json.loads((root/'manifest.json').read_text())
    for name, expected in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError('manifest path escapes root')
        if verify_files and sha(path)!=expected: raise ValueError('input file changed: '+name)
    return dict(manifest_sha256=digest,files=files)


def catalogue_options(frozen):
    """Read sealed GT-free catalogues; retain cut candidates beyond peer support."""
    root=Path(frozen); receipt=manifest(root,True)
    data=json.loads((root/'deployment.json').read_text())
    if data['schema']!='orphan-continuation-freeze-v1': raise ValueError('unexpected input catalogue')
    options=[]; skeleton=[]
    for member in data['members']:
        folder=root/member['path']; sub=manifest(folder,True)
        if sub['manifest_sha256']!=member['seal']: raise ValueError('member catalogue seal differs')
        catalog=json.loads((folder/'catalog.json').read_text())
        meta=dict(unit=member['unit'],member_index=member['member_index'],member=str(member['member']),
                  hive=member.get('hive',member.get('domain')),group=member['group'],clip=member.get('clip',member.get('video')))
        skeleton.append(meta)
        for e in catalog['edges']:
            if e['unit']!=meta['unit'] or e['member_index']!=meta['member_index']: raise ValueError('edge member changed')
            orphan_key=f"{meta['unit']}|{meta['member_index']}|orphan:{e['orphan_linked_id']}"
            options.append(dict(meta,action_id=e['edge_id'],type='extension',features=e['features10'],
                recipient=e['recipient_cut_join_key'],orphan_key=orphan_key,edge_id=e['edge_id'],
                parent_canonical_id=e['parent_canonical_id'],gap=e['gap'],max_error=e['normalized_error']))
        for c in catalog['cut_catalog']['candidates']:
            candidate_id=c.get('candidate_id',c.get('key'))
            if not candidate_id: raise ValueError('original197 global candidate id required')
            options.append(dict(meta,action_id='cut:'+candidate_id,type='cut',candidate_id=candidate_id,
                recipient=candidate_id,features=list(c['features_G'])+[float(c['delta']),float(c['direction']==1)],
                delta=c['delta'],direction=c['direction'],parent_canonical_id=c.get('parent_canonical_id',c['parent_id'])))
        # Raw rows, lineage and per-edge window arrays are not model inputs.
        del catalog
    validate(options)
    if len({(r['unit'],r['member']) for r in skeleton})!=len(skeleton): raise ValueError('duplicate member catalogue')
    return options,skeleton,dict(receipt,prereg=data['prereg'],source197_seal=data['source197_seal'],source195_seal=data['source195_seal'])


def load_source_labels(owners_root, heldout, options, *, source_domains, prereg, frozen_sha, domain_loader=None):
    """Metadata manifests may be read; ONLY three allowed payloads may be opened."""
    allowed=tuple(source_domains)
    if heldout not in DOMAINS or len(allowed)!=3 or set(allowed)!=set(DOMAINS)-{heldout}: raise ValueError('explicit other-three source domains required')
    info=manifest(owners_root); labels={}; receipts={}
    for hive in allowed:
        name=hive+'.json'
        if domain_loader is None:
            path=Path(owners_root)/name
            if sha(path)!=info['files'][name]: raise ValueError('source shard changed')
            data=json.loads(path.read_text())
        else: data=domain_loader(owners_root,hive)
        if data['domain']!=hive or data['prereg']!=prereg: raise ValueError('source shard scope differs')
        if data['frozen_manifest_sha256']!=frozen_sha: raise ValueError('source labels catalogue differs')
        found={}
        for unit in data['units']:
            if unit.get('domain',unit.get('hive'))!=hive: raise ValueError('source unit hive differs')
            for m in unit['members']:
                for r in [*m['edges'],*m['cuts']]:
                    key=r['action_id']; value=r['labels']['0.5']['class']
                    if key in found or value not in ('S','D','U'): raise ValueError('duplicate/invalid source label')
                    found[key]=value
        if set(found)!={r['action_id'] for r in options if r['hive']==hive}: raise ValueError('source catalogue differs')
        labels.update(found); receipts[hive]=dict(file=name,sha256=info['files'][name],options=len(found))
        del data, found
    return dict(source_hives=list(allowed),labels=labels,receipts=dict(owners_manifest_sha256=info['manifest_sha256'],source_shards=receipts,label_threshold='0.5'))


def field_train(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''): raise ValueError('full published prereg required')
    for expected in (args.frozen_sha,args.owners_sha):
        if not re.fullmatch('[0-9a-f]{64}',expected or ''): raise ValueError('pinned input seals required')
    if manifest(args.frozen)['manifest_sha256']!=args.frozen_sha: raise ValueError('frozen seal differs from GO')
    if manifest(args.owners)['manifest_sha256']!=args.owners_sha: raise ValueError('owners seal differs from GO')
    options,skeleton,fr=catalogue_options(args.frozen)
    if fr['prereg']!=args.prereg: raise ValueError('catalogue prereg differs')
    if len(skeleton)!=60 or {h:sum(r['hive']==h for r in skeleton) for h in DOMAINS}!=dict(zip(DOMAINS,(12,12,18,18))):
        raise ValueError('registered sixty-member four-domain inventory differs')
    groups=defaultdict(set)
    for m in skeleton:
        groups[m['unit']].add(m['member_index'])
        if not 0<=int(m['member'])<42: raise ValueError('registered old member ID required')
    if len(groups)!=20 or any(slots!={0,1,2} for slots in groups.values()): raise ValueError('twenty complete three-member units required')
    if fr['source195_seal']!='4f78f24d7fd13dc2eca9b206501a5326b4e42d476e1a9c4dd1ba98fe4bf61377' or fr['source197_seal']!='e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4':
        raise ValueError('fixed old195/197 catalogue required')
    if sum(r['type']=='cut' for r in options)!=1196: raise ValueError('full197 cut inventory required, no peer eligibility filter')
    def provider(target):
        output=load_source_labels(args.owners,target,options,source_domains=[h for h in DOMAINS if h!=target],prereg=args.prereg,frozen_sha=args.frozen_sha)
        print('source-only labels ready',target,len(output['labels']),flush=True)
        return output
    return train_and_seal(options,provider,args.out,dict(prereg=args.prereg,freeze_receipt=fr,
        owners_manifest_sha256=args.owners_sha),skeleton)


def self_test():
    rng=np.random.default_rng(20001)
    def option(hive, orphan, recipient, kind='extension'):
        key=f'{hive}|0|{recipient}|{orphan}' if kind=='extension' else f'{hive}|0|{recipient}'
        return dict(action_id=key,candidate_id=key,type=kind,features=rng.normal(size=DIMENSIONS[kind]).tolist(),hive=hive,
                    group=1,clip=hive+'clip',unit=hive+'unit',member='0',member_index=0,recipient=recipient,orphan_key=orphan)
    rows=[option(h,o,r) for h in DOMAINS for o in range(3) for r in range(2)]
    rows += [option(h,-1,r,'cut') for h in DOMAINS for r in range(2)]
    labels={r['action_id']: ('S' if r.get('orphan_key')==0 else 'D' if r.get('orphan_key') in (1,-1) else 'U') for r in rows}
    L=fit([r for r in rows if r['hive']!='35'],{r['action_id']:labels[r['action_id']] for r in rows if r['hive']!='35' and r['type']=='extension'},'L','35')
    J=fit([r for r in rows if r['hive']!='35'],{r['action_id']:labels[r['action_id']] for r in rows if r['hive']!='35'},'J','35')
    assert len(L['contexts'])==6 and len(J['contexts'])==6
    assert L['optimization']['max_abs_gradient']<1e-7 and J['optimization']['max_abs_gradient']<1e-7
    assert np.isclose(sum(c['weight'] for c in L['contexts']),1)
    assert all(np.isclose(sum(c['target']),1) for c in J['contexts'])
    assert all(len(c['omitted_U'])==1 for c in J['contexts'])
    # Independent old O(n²) expression, on an intentionally uneven hierarchy.
    uneven=[]
    for h in DOMAINS:
        for g in range(1,3 if h=='35' else 2):
            for v in range(2 if g==1 else 1):
                for i in range(1+v+g):
                    r=option(h,100*g+10*v+i,0)
                    r.update(action_id=f'{h}/{g}/{v}/{i}',unit=f'{h}/{g}/{v}',clip=f'{h}/{v}',group=g)
                    uneven.append(r)
    ulabels={r['action_id']:('S' if i%3 else 'D') for i,r in enumerate(uneven)}
    uc=training_contexts(uneven,ulabels,'L')
    for c in uc:
        h,g,v,k=c['hive'],c['group'],c['clip'],c['kind']
        old=1/(len({r['hive'] for r in uc})*
               len({r['group'] for r in uc if r['hive']==h})*
               len({r['clip'] for r in uc if (r['hive'],r['group'])==(h,g)})*
               len({r['kind'] for r in uc if (r['hive'],r['group'],r['clip'])==(h,g,v)})*
               sum((r['hive'],r['group'],r['clip'],r['kind'])==(h,g,v,k) for r in uc))
        assert c['weight']==old
    assert all(c['target']==[.5,.5,0.] for c in L['contexts'] if c['kind']=='positive')
    assert predict(json.loads(json.dumps(J)),rows)==predict(J,rows)
    ext=[r for r in rows if r['type']=='extension']
    perturbed=[dict(r,features=[1e30]*10) if labels[r['action_id']]=='U' else r for r in ext]
    elabels={r['action_id']:labels[r['action_id']] for r in ext}
    assert fit(ext,elabels)['heads']==fit(perturbed,elabels)['heads']
    assert fit(ext,elabels)['heads']==fit(list(reversed(ext)),elabels)['heads']
    empty=fit([],{},'J'); assert all(v is None for v in predict(empty,rows).values())
    # No source option type => explicit abstention, not fabricated coefficients.
    only=fit([r for r in rows if r['type']=='cut'],{r['action_id']:labels[r['action_id']] for r in rows if r['type']=='cut'},'J')
    assert only['heads']['extension']['status']=='empty'
    assert only['heads']['cut']['intercept']>0 and np.isfinite(only['heads']['cut']['intercept'])
    target=[r for r in rows if r['hive']=='35']; tiny=target[:4]+target[-2:]
    for _ in range(80):
        values=rng.integers(-2,4,len(tiny)).astype(float)
        scores=dict(zip([r['action_id'] for r in tiny],values))
        result=matching(tiny,scores)
        feasible=[]
        for bits in itertools.product((0,1),repeat=len(tiny)):
            chosen=[r for r,b in zip(tiny,bits) if b]
            if any(scores[r['action_id']]<=0 for r in chosen): continue
            if len({r['recipient'] for r in chosen})!=len(chosen): continue
            short=[r['orphan_key'] for r in chosen if r['type']=='extension']
            if len(set(short))!=len(short): continue
            ids=sorted(r['action_id'] for r in chosen)
            feasible.append((sum(Fraction(scores[k]) for k in ids),tuple(k in ids for k in sorted(scores)),ids))
        assert sum(Fraction(scores[k]) for k in result)==max(t[0] for t in feasible)
    # Alternating augmentation beats greedy; zeros never enter the matching.
    a,b,c=[option('35',o,r) for o,r in ((0,0),(1,0),(0,1))]
    assert matching([a,b,c],{a['action_id']:3.,b['action_id']:2.,c['action_id']:2.})==sorted([b['action_id'],c['action_id']])
    for i in range(5):
        perm=rng.permutation(len(target)); reordered=[target[j] for j in perm]
        scores={r['action_id']:1. for r in target}
        assert matching(target,scores)==matching(reordered,scores)
    singleton=[option('35',0,0)]
    score={singleton[0]['action_id']:2.}; assert shuffled_scores(singleton,score)[0]==score
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp); owners=temp/'owners'; owners.mkdir()
        frozen=temp/'frozen'; folder=frozen/'u00'; folder.mkdir(parents=True)
        parent='35_g1_clip|0|7'
        save(folder/'catalog.json',dict(edges=[dict(unit='35_g1_clip',member_index=0,edge_id='edge',
            recipient_cut_join_key=parent,orphan_linked_id=91,parent_canonical_id=7,gap=3,
            normalized_error=.2,features10=[.25,.5,0.,.1,.2,.3,0.,0.,0.,1.])],
            cut_catalog=dict(candidates=[dict(candidate_id=parent,parent_id=7,features_G=[0.]*9,delta=-1,direction=1)])))
        save(folder/'manifest.json',{'catalog.json':sha(folder/'catalog.json')})
        (folder/'SEALED.sha256').write_text(sha(folder/'manifest.json'))
        save(frozen/'deployment.json',dict(schema='orphan-continuation-freeze-v1',prereg='synthetic',source197_seal='197',source195_seal='195',
            members=[dict(unit='35_g1_clip',member_index=0,member='0',hive='35',group=1,clip='clip',path='u00',seal=sha(folder/'manifest.json'))]))
        save(frozen/'manifest.json',{'deployment.json':sha(frozen/'deployment.json')})
        (frozen/'SEALED.sha256').write_text(sha(frozen/'manifest.json'))
        projected,skel,_=catalogue_options(frozen)
        assert len(projected)==2 and len(skel)==1 and projected[0]['recipient']==projected[1]['recipient']
        assert projected[1]['features']==[0.]*9+[-1.,1.]
        assert len(matching(projected,{'edge':1.,'cut:'+parent:2.}))==1
        for h in DOMAINS:
            source=[r for r in rows if r['hive']==h]
            m={k:[dict(action_id=r['action_id'],labels={'0.5':{'class':labels[r['action_id']]}}) for r in source if r['type']==kind]
               for k,kind in (('edges','extension'),('cuts','cut'))}
            save(owners/(h+'.json'),dict(domain=h,prereg='synthetic',frozen_manifest_sha256='frozen',units=[dict(domain=h,members=[m])]))
        save(owners/'manifest.json',{p.name:sha(p) for p in owners.iterdir()})
        (owners/'SEALED.sha256').write_text(sha(owners/'manifest.json'))
        # Each fold's heldout owner file is physically absent, not just unused.
        for h in DOMAINS:
            path=owners/(h+'.json'); raw=path.read_bytes(); path.unlink()
            loaded_labels=load_source_labels(owners,h,rows,source_domains=[d for d in DOMAINS if d!=h],prereg='synthetic',frozen_sha='frozen')
            assert set(loaded_labels['labels'])=={r['action_id'] for r in rows if r['hive']!=h}
            path.write_bytes(raw)
        spy=[]
        try:
            load_source_labels(owners,'35',rows,source_domains=['35','W','G'],prereg='synthetic',frozen_sha='frozen',domain_loader=lambda *a:spy.append(a))
        except ValueError: pass
        else: raise AssertionError('invalid whitelist accepted')
        assert not spy
        calls=[]
        def provider(h):
            calls.append(h); allowed=[d for d in DOMAINS if d!=h]
            return dict(source_hives=allowed,labels={r['action_id']:labels[r['action_id']] for r in rows if r['hive'] in allowed})
        result=train_and_seal(rows,provider,temp/'pred',dict(prereg='synthetic'))
        loaded=load_predictions(result['root']); assert len(calls)==4 and len(loaded['decisions'])==4
    return dict(status='PASS',seed=20001,matching_exhaustive_cases=80,tests='known/U/null targets; weight1; gradients; empty type; JSON replay; float matching versus exhaustive; greedy counterexample; ID tie/order; M singleton; all4 heldout shard absent; whitelist rejects before spy; common fourfold seal')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--run-field',action='store_true'); parser.add_argument('--prereg')
    parser.add_argument('--frozen',type=Path); parser.add_argument('--owners',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--frozen-sha');parser.add_argument('--owners-sha')
    args=parser.parse_args()
    if args.run_field:
        if any(v is None for v in (args.frozen,args.owners,args.out)): parser.error('--frozen/--owners/--out required')
        print(json.dumps(field_train(args),indent=2))
    else: print(json.dumps(self_test(),indent=2))
