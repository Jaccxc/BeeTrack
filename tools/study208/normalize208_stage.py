"""Explicit future208 projection. No live-file reads or imports on startup.

The caller needs the coordinator's numeric token and a pinned final independent
receipt. It never loads crops.npy, image files, GT rows or tensor weights.
"""
import argparse
import hashlib
import json
import math
import gzip
from pathlib import Path


def normalize(root, out, independent_path, independent_sha, diagnostic_index):
    import sys
    sys.path.insert(0, str(Path.cwd()))
    import numpy as np
    from beetrack.evaluate import inside
    from prepare208_release import (PREREG, SCHEMA, file_sha, public_numeric,
                                    project_K_catalog, project_native_state, project_summary)
    from replay208 import ARMS, CLIPS, select_cuts, displaced_scores
    from tools.study202.stream202 import Reader
    root, out = Path(root).resolve(), Path(out)
    assert file_sha(independent_path) == independent_sha
    independent = json.loads(Path(independent_path).read_text(encoding='utf-8'))
    assert independent['passed'] and independent['prereg'] == PREREG
    seals = independent['input_seals']
    manifests = {}
    for stage in ('inputs', 'labels', 'predictions', 'score'):
        assert file_sha(root/stage/'manifest.json') == seals[stage]
        assert (root/stage/'SEALED.sha256').read_text().strip() == seals[stage]
        manifests[stage] = json.loads((root/stage/'manifest.json').read_text(encoding='utf-8'))
    sources = {}

    def checked(stage, name):
        path = (root/stage/name).resolve()
        assert path.is_relative_to(root/stage) and file_sha(path) == manifests[stage][name]
        sources[stage+'/'+name] = manifests[stage][name]
        return path

    def read(stage, name):
        return json.loads(checked(stage, name).read_text(encoding='utf-8'))

    top, forecasts = read('inputs', 'deployment.json'), read('predictions', 'metadata.json')
    assert top['prereg'] == forecasts['prereg'] == PREREG
    summary = read('score', 'summary.json')
    assert summary['physical_scope']['state_requests'] == 54
    truth_index = json.loads(Path(diagnostic_index).read_text(encoding='utf-8'))
    assert truth_index['passed'] and truth_index['main_sha256'] == independent_sha
    truth_by_clip = {c['clip']: c for c in truth_index['clips']}
    out.mkdir(parents=True, exist_ok=False)
    contract = dict(schema=SCHEMA, prereg=PREREG, source_seals=seals,
                    verification_sha256=independent_sha, clips=[], folds=[], components=[],
                    scope=dict(contains_pixels=False, contains_raw_GT=False,
                               CNN_refit_from_public_numeric_verified=False))

    def emit(role, key, value):
        public_numeric(value)
        token = hashlib.sha256((role+'|'+key).encode()).hexdigest()+'.json'
        path = out/token
        encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
        with path.open('w', encoding='utf-8', newline='\n') as stream:
            for chunk in encoder.iterencode(value):
                stream.write(chunk)
        identity = role+'|'+key
        contract['components'].append(dict(id=identity, role=role, key=key,
                                           file=token, sha256=file_sha(path)))
        return identity

    for heldout, fold in forecasts['folds'].items():
        target = dict(heldout=heldout, source_domains=fold['source_domains'],
                      training_seed=208, models={})
        for arm, model in fold['models'].items():
            cal = model['calibration']
            with np.load(checked('predictions', cal['scores_file']), allow_pickle=False) as bank:
                values = []
                for probability, clip_index, bag_index, cut_index in zip(
                        bank['probabilities'], bank['clip_index'], bank['bag_index'], bank['cut_index']):
                    clip_key = cal['clip_order'][int(clip_index)]
                    hive = clip_key.split('/')[0]
                    assert hive != heldout and math.isfinite(float(probability))
                    values.append(dict(hive=hive, clip=clip_key, cut_index=int(cut_index),
                                       bag_index=int(bag_index), kind='exact_survival',
                                       length=1, valid=True, score=float(probability)))
            cal_id = emit('calibration', heldout+'/'+arm, dict(negative_records=values))
            training = read('predictions', model['training_file'])
            training.pop('source_receipt', None)
            training.pop('calibration', None)
            emit('training_receipt', heldout+'/'+arm, training)
            target['models'][arm] = dict(status=model['status'], threshold=model['threshold'],
                calibration_component=cal_id, model_weight_sha256=manifests['predictions'][model['model_file']],
                registered_parameters=model['registered_parameters'],
                structural_gradient_parameter_limit=model['structural_gradient_parameter_limit'])
        contract['folds'].append(target)
    pred_index = {(c['domain'], c['clip']): c for c in forecasts['clips']}
    assert {(c['domain'], c['clip'], c['seed']) for c in top['clips']} == set(CLIPS)
    for ref in top['clips']:
        hive, clip, prefix = ref['domain'], ref['clip'], ref['path']
        key = hive+'/'+clip
        meta = read('inputs', prefix+'/metadata.json')
        rows = np.load(checked('inputs', prefix+'/rows.npy'), mmap_mode='r', allow_pickle=False)
        canon = np.load(checked('inputs', prefix+'/canon.npy'), mmap_mode='r', allow_pickle=False)
        indices = np.load(checked('inputs', prefix+'/indices.npy'), mmap_mode='r', allow_pickle=False)
        valid = np.load(checked('inputs', prefix+'/valid.npy'), mmap_mode='r', allow_pickle=False)
        keys = read('inputs', prefix+'/source_keys.json')
        side = inside(rows, meta['polygon'])
        # First build only run endpoints; all CUT metadata remains available.
        catalog = project_K_catalog(rows, canon, indices, valid, side, keys, selected_cut_union=())
        outputs, selected_union = {}, set()
        for arm in ARMS:
            files = pred_index[hive, clip]['outputs'][arm]
            raw = np.load(checked('predictions', files['probabilities']), mmap_mode='r', allow_pickle=False)
            probabilities = [float(p) if math.isfinite(float(p)) else None for p in raw]
            selected = np.load(checked('predictions', files['selected']), allow_pickle=False).tolist()
            raw05 = np.load(checked('predictions', files['raw05']), allow_pickle=False).tolist()
            model = forecasts['folds'][hive]['models'][arm]
            active, threshold = model['status'] == 'trained', model['threshold']
            selected_union.update(select_cuts(catalog['cuts'], probabilities, threshold, active=active))
            if arm == 'relative_cross_interval':
                shifted = displaced_scores(catalog['cuts'], probabilities)
                selected_union.update(select_cuts(catalog['cuts'], shifted, threshold, active=active))
            outputs[arm] = dict(probabilities=probabilities, selected=selected,
                                raw05=raw05, threshold=threshold)
        # Coordinates are needed only at actually selected boundaries and run ends.
        for number in selected_union:
            cut = catalog['cuts'][number]
            for raw_index in (cut['left_raw'], cut['right_raw']):
                row = rows[raw_index]
                catalog['endpoints'][str(raw_index)] = dict(frame=int(row[0]), inside=bool(side[raw_index]),
                    center=[float(row[2]+row[4]/2), float(row[3]+row[5]/2)],
                    diagonal=float(np.hypot(row[4], row[5])), source_key=list(keys[raw_index]))
        entry = dict(domain=hive, clip=clip, seed=ref['seed'], nframes=meta['nf'],
                     catalog_component=emit('catalog', key, catalog),
                     prediction_component=emit('predictions', key, outputs))
        diagnostic = truth_by_clip[key]['files']['adjacency.jsonl.gz']
        path = (Path(diagnostic_index).parent/diagnostic['path']).resolve()
        assert path.is_relative_to(Path(diagnostic_index).parent.resolve())
        assert file_sha(path) == diagnostic['sha256']
        order = np.lexsort((np.arange(len(rows)), rows[:, 0], canon))
        starts = np.r_[True, np.diff(canon[order]) != 0] if len(rows) else np.empty(0, dtype=bool)
        offsets = np.maximum.accumulate(np.where(starts, np.arange(len(rows)), 0))
        ordinals = np.empty(len(rows), dtype=np.int64)
        ordinals[order] = np.arange(len(rows))-offsets

        def normalized_mother():
            if path.name.endswith('.jsonl.gz'):
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        row['run_key'] = int(row['run_key'].rsplit('|p', 1)[1])
                        yield row
                return
            with path.open(encoding='utf-8') as stream:
                reader = Reader(stream)
                def normalized(row):
                    if 'candidate' not in row:
                        row['candidate'] = row.pop('input_ordinal')
                    raw = row['right_raw']
                    assert row['right_frame'] == int(rows[raw, 0])
                    row['right_ordinal'] = int(ordinals[raw])
                    row['run_key'] = int(canon[raw])
                    return row
                if diagnostic.get('array_field', 'adjacency_mother') is None:
                    for _ in reader.array():
                        yield normalized(reader.value())
                else:
                    found = False
                    for field in reader.object():
                        if field != diagnostic.get('array_field', 'adjacency_mother'):
                            reader.skip_value()
                            continue
                        found = True
                        for _ in reader.array():
                            yield normalized(reader.value())
                    assert found
                assert not reader.peek()
        chunks, buffer = [], []
        for row in normalized_mother():
            buffer.append(row)
            if len(buffer) == 8192:
                chunks.append(emit('diagnostic_rows', key+'/'+str(len(chunks)), buffer))
                buffer.clear()
        if buffer:
            chunks.append(emit('diagnostic_rows', key+'/'+str(len(chunks)), buffer))
        entry['diagnostic_component'] = emit('diagnostics', key, dict(adjacency_components=chunks))
        del order, starts, offsets, ordinals, buffer
        states = {}
        for state in next(c for c in summary['clips'] if c['domain'] == hive and c['clip'] == clip)['states']:
            relative = 'clips/'+key+'/'+state+'/score.json'
            record = read('score', relative)
            states[state] = project_native_state(record, sources['score/'+relative])
            del record
        entry['state_component'] = emit('states', key, states)
        contract['clips'].append(entry)
        del rows, canon, indices, valid, side, keys, catalog, outputs, states
    contract['summary_component'] = emit('expected_summary', 'all', project_summary(summary))
    source = Path(__file__).with_name('verify208.py')
    assert file_sha(source) == independent['sources']['verify208.py']
    normalized_source = source.read_bytes().replace(b'\r\n', b'\n')
    (out/'verify208.py').write_bytes(normalized_source)
    contract['diagnostic_source_sha256'] = hashlib.sha256(normalized_source).hexdigest()
    contract['original_diagnostic_source_sha256'] = file_sha(source)
    contract['source_files'] = sources
    public_numeric(contract)
    (out/'contract.json').write_text(json.dumps(contract, sort_keys=True, indent=2)+'\n',
                                    encoding='utf-8', newline='\n')
    return contract


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--independent', type=Path, required=True)
    p.add_argument('--independent-sha', required=True)
    p.add_argument('--diagnostic-index', type=Path, required=True)
    p.add_argument('--numeric-go', action='store_true', required=True)
    args = p.parse_args()
    normalize(args.root, args.out, args.independent, args.independent_sha, args.diagnostic_index)
