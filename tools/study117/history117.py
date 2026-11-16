"""Recover the six §117 historical fixed-640 P0 controls from the public §108 source archive."""
import gzip
import hashlib
import json
from pathlib import Path

from tools.study115.graph115 import Graph, _events, _grids, _window


SOURCE_SHA256 = '2a7d8bf2d4b027dafa59d924e5b908ff2b88c0edbdf9fcded097c60383edd383'
CLIPS = {'BEE24-35-tail': 33, 'BEE24-33-tail': 35}


def metrics(predictions, labels):
    n = labels['nframes']
    pred = _events([row[:2] for row in predictions], n)
    gt = _events(labels['gt'], n)
    gt_counts = {d: sum(sign*f > 0 for f in gt.values()) for d,sign in (('in', 1), ('out', -1))}
    pred_counts = {d: sum(sign*f > 0 for f in pred.values()) for d,sign in (('in', 1), ('out', -1))}
    pred_bins = _grids(pred.values(), n)
    score, proof = _window(pred_bins, _grids(gt.values(), n), 1, [gt_counts['in'], gt_counts['out']])
    terms = proof['abs_error_in_out_by_offset']
    window = dict(width_frames=750, fps=25, offsets_frames=list(range(0,750,25)),
                  bins_per_offset=[len(bins) for bins in pred_bins], l1_by_offset_in_out=terms,
                  numerator=sum(map(sum,terms)), denominator=30*len(gt))
    iou = {}
    for tau, key in (('0.5','edges05'), ('0.3','edges03')):
        result = Graph(pred, gt, labels[key]).score()
        for counts in result['by_direction'].values():
            tp, fp, fn = (counts[k] for k in ('tp','fp','fn'))
            counts['f1'] = 2*tp/max(1, 2*tp+fp+fn)
        iou[tau] = result
    return dict(w30_percent=score['w_percent'], window=window, gt_counts=gt_counts, pred_counts=pred_counts,
                bias={d: pred_counts[d]-gt_counts[d] for d in gt_counts}, iou=iou)


def source_rows(bundle, source_clip_specs):
    masks = bundle['masks']
    if masks['role'] != 'source' or set(source_clip_specs) != set(CLIPS):
        raise ValueError('Expected both registered source-tail clips and the §108 source archive')
    cases = {}
    for clip, fold in CLIPS.items():
        spec = source_clip_specs[clip]
        if isinstance(spec, dict):
            if spec.get('nframes', 1000) != 1000 or list(spec.get('seeds', range(3))) != [0,1,2]:
                raise ValueError('Clip clock or seed roster differs from the registered source controls')
            spec = spec['fold']
        if spec != fold:
            raise ValueError('Mismatched source clip and held-out detector fold')
        by_seed = cases[clip] = {}
        first_truth = None
        for seed in range(3):
            key = f'{fold}_{seed}_tail'
            tag = f'loho_no{fold}_s{seed}'
            prediction = masks['predictions'][key]
            label = bundle['labels'][key]
            expected = {'fold': fold, 'seed': seed, 'clip': clip, 'nframes': 1000, 'split': 'val', 'tag': tag}
            if any(row[k] != v for row in (prediction,label) for k,v in expected.items()):
                raise ValueError(f'§108 prediction or GT rows have the wrong source: {key}')
            pred_hash = masks['prediction_sha256'][key]
            if label['predictions_sha256'] != pred_hash:
                raise ValueError(f'§108 graph was made from different predictions: {key}')
            truth = (label['gt_mot_sha256'], label['gt'])
            if first_truth is not None and truth != first_truth:
                raise ValueError(f'Historical GT differs across detector seeds: {clip}')
            first_truth = truth
            inputs = prediction['sources']
            evidence = {k: label[k] for k in ('nframes', 'gt', 'edges05', 'edges03')}
            evidence['predictions'] = prediction['predictions']
            by_seed[str(seed)] = dict(clip=clip, fold=fold, seed=seed, mot_sha256=prediction['mot_sha256'],
                gt_mot_sha256=label['gt_mot_sha256'], detector_sha256=inputs[f'dets/{tag}/val/{clip}.npy'],
                strict_linker_sha256=inputs['m_dev_strict.json'], zones_sha256=inputs['beetrack/zones.json'],
                source_record_sha256=pred_hash, source_label_sha256=bundle['label_sha256'][key],
                evidence=evidence, **metrics(evidence['predictions'], evidence))
    return cases


def history(archive, source_clip_specs=None):
    """Source specs are {clip: held-out fold} or {clip: {fold, nframes, seeds}}. W is in percent."""
    raw = Path(archive).read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != SOURCE_SHA256:
        raise ValueError('Expected the original public docs/study108_source.json.gz')
    bundle = json.loads(gzip.decompress(raw))
    cases = source_rows(bundle, CLIPS if source_clip_specs is None else source_clip_specs)
    return dict(archive_sha256=checksum, method='historical fixed-640 native P0', cases=cases)
