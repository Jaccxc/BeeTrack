"""Explicit CPU-only source mining experiment; imports alone never read field data."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

PRIOR_SHA = 'ce382bc2b1b1a986db990a5d68a5eb2b696cfa39'
ARMS = ('H', 'U', 'base')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def modules(repo):
    sys.path.insert(0, str(repo / 'tools/study179'))
    sys.path.insert(0, str(repo))
    names = ('hard_negative', 'strip_model', 'strip_data', 'strip_score', 'predict_events', 'run_boundary')
    return SimpleNamespace(**{name: importlib.import_module(name) for name in names})


def code_hashes(m):
    paths = [Path(__file__), *[Path(getattr(m, n).__file__) for n in vars(m)]]
    return {p.name: hashlib.sha256(p.read_text(encoding='utf-8').encode('utf-8')).hexdigest() for p in paths}


def prior_source(args, m):
    """Read only this source's prior checkpoint, receipts, head and tail caches."""
    folder = args.prior / f'fold_source{args.source}'
    receipt = read(folder / 'receipt.json')
    if receipt['prereg'] != PRIOR_SHA or receipt['source'] != args.source:
        raise ValueError('wrong prior registration/source')
    for name in ('strip_data.py', 'strip_model.py', 'strip_score.py', 'run_boundary.py'):
        if receipt['code'][name] != m.run_boundary.code_hashes()[name]:
            raise ValueError(f'prior code mismatch: {name}')
    checkpoint = folder / 'T_state.pt'
    if digest(checkpoint) != receipt['files']['T_state.pt']:
        raise ValueError('prior checkpoint hash mismatch')
    import torch
    bundle = args.repo / f'tools/study179/weights/source{args.source}_T.npz'
    base, arm, threshold = m.predict_events.load_bundle(bundle)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if arm != 'T' or threshold != receipt['arms']['T']['threshold']:
        raise ValueError('public bundle metadata mismatch')
    if set(state) != set(base.state_dict()) or any(not torch.equal(v, base.state_dict()[k]) for k, v in state.items()):
        raise ValueError('public numeric bundle differs from prior source checkpoint')
    cache_args = SimpleNamespace(out=args.prior, repo=args.repo, prereg=PRIOR_SHA)
    head, head_receipt = m.run_boundary.load_cache(cache_args, args.source, 'head')
    tail, tail_receipt = m.run_boundary.load_cache(cache_args, args.source, 'tail')
    if head_receipt != receipt['source_head'] or tail_receipt != receipt['source_tail']:
        raise ValueError('prior cache receipt mismatch')
    if len(tail) != 1000:
        raise ValueError('expected exactly 1000 source-tail frames')
    provenance = dict(prior_prereg=PRIOR_SHA, prior_receipt_sha256=digest(folder / 'receipt.json'),
                      public_bundle_sha256=digest(bundle), checkpoint_sha256=digest(checkpoint),
                      source_head_gt_sha256=receipt['source_head_gt_sha256'],
                      source_tail_gt_sha256=receipt['source_tail_gt_sha256'],
                      source_head=head_receipt, source_tail=tail_receipt)
    return base, head, tail, provenance


def freeze(args, m):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    destination = args.out / f'fold_source{args.source}'
    if destination.exists():
        raise FileExistsError(destination)
    base, head, tail, provenance = prior_source(args, m)
    head_gt, head_sha = m.run_boundary.truth(args, args.source, 'head')
    # Only calibration labels survive this expression; half2 is not scored here.
    tail_events, tail_sha = m.run_boundary.truth(args, args.source, 'tail')
    cal_gt = [e for e in tail_events if 1 <= e[0] <= 488]
    del tail_events
    if head_sha != provenance['source_head_gt_sha256'] or tail_sha != provenance['source_tail_gt_sha256']:
        raise ValueError('source GT changed since prior registration')
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'freeze{args.source}-', dir=args.out) as temporary:
        stage = Path(temporary) / 'fold'
        stage.mkdir()
        arms = {}
        for arm in ARMS:
            audit = []
            if arm == 'base':
                model = base
            else:
                model, audit = m.hard_negative.train_model(head, head_gt, 1, len(head),
                    base.state_dict(), mode='hard' if arm == 'H' else 'uniform', model_module=m.strip_model)
            first = m.strip_model.predict_model(model, tail, 1, 488, 'T')
            calibration = m.strip_score.calibrate(first, cal_gt)
            threshold = calibration['threshold']
            np.save(stage / f'{arm}_half1_scores.npy', first, allow_pickle=False)
            write(stage / f'{arm}_half1_metrics.json', calibration)
            second = m.strip_model.predict_model(model, tail, 513, 1000, 'T')
            np.save(stage / f'{arm}_half2_scores.npy', second, allow_pickle=False)
            write(stage / f'{arm}_half2_events.json', m.strip_score.decode(second, threshold, start=513))
            # Direct safe numeric export; load_bundle validates all shapes/dtypes.
            arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            code, percent = m.predict_events._metadata('T', threshold)
            np.savez_compressed(stage / f'{arm}_weights.npz', **arrays, arm_code=code, threshold_percent=percent)
            m.predict_events.load_bundle(stage / f'{arm}_weights.npz')
            arms[arm] = dict(threshold=threshold, calibration=calibration, audit=audit)
        files = {p.name: digest(p) for p in stage.iterdir()}
        receipt = dict(prereg=args.prereg, source=args.source, code=code_hashes(m), provenance=provenance,
                       head_gt_sha256=head_sha, cal_interval=[1,488], held_interval=[513,1000],
                       arms=arms, files=files)
        write(stage / 'receipt.json', receipt)
        stage.rename(destination)
    return dict(source=args.source, sealed=True, receipt_sha256=digest(destination / 'receipt.json'))


def frozen(args, m, source):
    folder = args.out / f'fold_source{source}'
    receipt = read(folder / 'receipt.json')
    if receipt['prereg'] != args.prereg or receipt['source'] != source or receipt['code'] != code_hashes(m):
        raise ValueError('freeze identity/code mismatch')
    if receipt['cal_interval'] != [1,488] or receipt['held_interval'] != [513,1000]:
        raise ValueError('interval mismatch')
    for name, sha in receipt['files'].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError('frozen artifact mismatch')
    return folder, receipt


def score(args, m):
    import numpy as np
    destination = args.out / 'summary.json'
    if destination.exists():
        raise FileExistsError(destination)
    folds = [frozen(args, m, source) for source in (33,35)]
    summary = dict(prereg=args.prereg, metric='time-cell F1; not official box F1', folds=[])
    # Both folds and every artifact have been verified before any held GT access.
    for folder, receipt in folds:
        source = receipt['source']
        all_gt, gt_sha = m.run_boundary.truth(args, source, 'tail')
        if gt_sha != receipt['provenance']['source_tail_gt_sha256']:
            raise ValueError('source tail GT changed since freeze')
        gt = [e for e in all_gt if 513 <= e[0] <= 1000]
        row = dict(source=source, gt_events=gt, gt_sha256=gt_sha,
                   receipt_sha256=digest(folder / 'receipt.json'), arms={})
        for arm in ARMS:
            threshold = receipt['arms'][arm]['threshold']
            cube = np.load(folder / f'{arm}_half2_scores.npy', allow_pickle=False)
            if cube.shape != (488,32,2):
                raise ValueError('held cube shape mismatch')
            events = read(folder / f'{arm}_half2_events.json')
            if events != [list(e) for e in m.strip_score.decode(cube, threshold, start=513)]:
                raise ValueError('decode mismatch')
            result = m.strip_score.counts(events, gt)
            for d in ('in','out'):
                r = result[d]
                r.update(fp=r['pred']-r['tp'], fn=r['gt']-r['tp'])
            result.update(fp=result['pred']-result['tp'], fn=result['gt']-result['tp'], events=events,
                          err=m.strip_score.window_err(events, gt, 488, start=513), threshold=threshold)
            result['W'] = 100*result['err']/len(gt) if gt else None
            result['zero_gt'] = not bool(gt)
            result['inadequate_direction_gt'] = [d for d in ('in','out') if result[d]['gt']==0]
            row['arms'][arm] = result
        summary['folds'].append(row)
    with tempfile.TemporaryDirectory(prefix='score-', dir=args.out) as temporary:
        path = Path(temporary)/'summary.json'
        write(path, summary)
        path.rename(destination)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','score'))
    parser.add_argument('--source', type=int, choices=(33,35))
    for name in ('repo','prior','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--prereg', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.prereg) or args.prereg == PRIOR_SHA:
        parser.error('new full registration SHA required')
    if args.command == 'freeze' and args.source is None:
        parser.error('freeze requires --source')
    for name in ('repo','prior','out'):
        setattr(args, name, getattr(args,name).resolve())
    if args.out == args.prior:
        parser.error('out must differ from prior')
    m = modules(args.repo)
    result = freeze(args,m) if args.command == 'freeze' else score(args,m)
    if args.command == 'score':
        for fold in result['folds']:
            for arm,r in fold['arms'].items():
                print(fold['source'],arm,{k:r[k] for k in ('tp','fp','fn','f1','W','in','out','zero_gt','inadequate_direction_gt')})
    else:
        print(json.dumps(result))


if __name__ == '__main__':
    main()
