"""Source-only incremental208 package boundary. Never discover live field data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
import shutil

PREREG = 'c358042292d912cbf74e21859f6e0fba45ad1cf8'
SCHEMA = 'switch208-public-numeric-contract-v1'
FORBIDDEN = {'pixels', 'crops', 'images', 'image_bytes', 'GT_rows', 'gt_rows',
             'original_gt_id', 'original_GT_ID', 'GT_run_key', 'owners_raw', 'GT_boxes'}


def project_native_state(record, source_sha256):
    """Official event sufficient statistics, with all individual GT IDs removed.

    A graph replay can verify its cardinality, not regenerate frame IoUs. Raw
    GT coordinates, dates and identities are never copied from native evidence.
    """
    assert re.fullmatch('[0-9a-f]{64}', source_sha256)
    metric = record['metrics']
    evidence = metric['evidence']
    truth = [int(row[0]) for row in evidence['gt']]
    prediction = [int(row[0]) for row in evidence['predictions']]
    assert len(set(truth)) == len(truth) == metric['nG']
    assert len(set(prediction)) == len(prediction) == metric['nP']
    gi, pi = {v: i for i, v in enumerate(truth)}, {v: i for i, v in enumerate(prediction)}
    out = dict(native_score_json_sha256=source_sha256,
               predicted_signed_dates=[int(row[1]) for row in evidence['predictions']],
               pred_bins=metric['pred_bins'], truth_window_counts=metric['gt_bins'],
               nGT=metric['nG'], opaque_edges={}, official={},
               numerator=record['numerator'], denominator=record['denominator'],
               raw_row_bijection=record['raw_row_bijection'], physical_rows=record['physical_rows'],
               frame_IoU_recomputed_from_public_numeric=False)
    for threshold, key in (('0.5', 'edges05'), ('0.3', 'edges03')):
        out['opaque_edges'][threshold] = [[pi[int(p)], gi[int(g)]] for p, g in evidence[key]]
        out['official'][threshold] = {k: metric['iou'][threshold][k] for k in ('tp', 'pred', 'gt', 'f1')}
    proof = record['proof']
    out['deployment'] = {k: proof[k] for k in ('nms', 'displacement') if k in proof}
    if 'cut' in proof:
        cut = proof['cut']
        out['deployment']['cut'] = dict(selected=cut['selected'], cuts=cut['cuts'],
            segments=[dict({k: v for k, v in s.items() if k != 'raw_indices'},
                           nrows=len(s['raw_indices'])) for s in cut['segments']])
    if proof.get('relink') is not None:
        out['deployment']['relink'] = proof['relink']
    public_numeric(out)
    return out


def project_summary(summary):
    assert summary['schema'] == 'switch208-native-stage-v1'
    out = dict(domains=summary['domains'],
               registered_gate={k: summary['registered_gate'][k] for k in ('checks', 'all_checks')},
               physical_scope=summary['physical_scope'])
    public_numeric(out)
    return out


def project_frame_matching(frame, ious, expected_owner_positions, order_hashes):
    """Optional numeric IoU-cost witness; opaque within-frame row positions.

    This accepts an already computed matrix. It never reads GT or calculates
    geometrical IoUs, and does not assert that such witnesses will be available.
    """
    nr = len(ious)
    nc = len(ious[0]) if nr else len(expected_owner_positions['0.5'])
    assert all(len(row) == nc for row in ious)
    assert all(math.isfinite(float(v)) and 0 <= v <= 1 for row in ious for v in row)
    assert all(len(expected_owner_positions[t]) == nc for t in ('0.5', '0.3'))
    out = dict(frame=frame, gt_row_count=nr, pred_row_count=nc,
               iou_nonzero=[[g, p, float(v)] for g, row in enumerate(ious)
                            for p, v in enumerate(row) if v],
               opaque_owner_positions=expected_owner_positions, order_hashes=order_hashes,
               interpretation='Given IoU costs and row-order fingerprints; no raw GT identity/geometry')
    public_numeric(out)
    return out


def project_K_catalog(rows, canonical, indices, valid, inside_flags, source_keys,
                      selected_cut_union=None):
    """Future mmap caller supplies K geometry/side flags; no pixels/GT needed."""
    from collections import defaultdict
    import numpy as np
    assert len(rows) == len(canonical) == len(inside_flags) == len(source_keys)
    assert len(indices) == len(valid)
    needed_cuts = None if selected_cut_union is None else set(selected_cut_union)
    members = defaultdict(list)
    for i, parent in enumerate(canonical):
        members[int(parent)].append(i)
    runs, endpoints, cuts, positions = [], {}, [], {}
    wanted = set()
    for parent, ii in sorted(members.items()):
        ii.sort(key=lambda i: (rows[i][0], i))
        assert all(0 < rows[b][0] - rows[a][0] <= 25 for a, b in zip(ii, ii[1:]))
        for ordinal, raw in enumerate(ii):
            positions[raw] = parent, ordinal
        changes = [dict(ordinal=j, frame=int(rows[raw][0]), inside=bool(inside_flags[raw]))
                   for j, raw in enumerate(ii) if j and inside_flags[raw] != inside_flags[ii[j-1]]]
        runs.append(dict(run_key=parent, nrows=len(ii), first_raw=ii[0], last_raw=ii[-1], side_changes=changes))
        wanted.update((ii[0], ii[-1]))
    for number, quad in enumerate(indices):
        quad = [int(i) for i in quad]
        parent, ordinal = positions[quad[2]]
        assert [positions[i] for i in quad] == [(parent, j) for j in range(ordinal-2, ordinal+2)]
        assert all(1 <= rows[b][0] - rows[a][0] <= 3 for a, b in zip(quad, quad[1:]))
        cuts.append(dict(candidate=number, run_key=parent, right_ordinal=ordinal,
                         left_raw=quad[1], right_raw=quad[2], left_frame=int(rows[quad[1]][0]),
                         right_frame=int(rows[quad[2]][0]), valid=bool(valid[number]),
                         source_keys=[list(source_keys[i]) for i in quad]))
        # Final packaging can retain endpoint coordinates only for cuts actually
        # selected by one of the sealed controls; every CUT still keeps its key.
        if needed_cuts is None or number in needed_cuts:
            wanted.update((quad[1], quad[2]))
    for raw in sorted(wanted):
        row = rows[raw]
        assert row[4] > 0 and row[5] > 0
        endpoints[str(raw)] = dict(frame=int(row[0]), inside=bool(inside_flags[raw]),
            center=[float(row[2]+row[4]/2), float(row[3]+row[5]/2)],
            diagonal=float(np.hypot(row[4], row[5])), source_key=list(source_keys[raw]))
    result = dict(runs=runs, endpoints=endpoints, cuts=cuts, native_rows=len(rows))
    public_numeric(result)
    return result


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def public_numeric(value):
    """Allow numeric summaries/opaque labels, reject raw material/private paths."""
    if isinstance(value, dict):
        assert not (set(value) & FORBIDDEN)
        for key, item in value.items():
            public_numeric(key)
            public_numeric(item)
    elif isinstance(value, list):
        for item in value:
            public_numeric(item)
    elif isinstance(value, str):
        assert not re.search(r'(?i)(?:[A-Z]:[\\/]|/Users/|/home/|AppData|_local[\\/]|data:image/)', value)
    elif type(value) is float:
        assert math.isfinite(value)
    elif value is not None and type(value) not in (bool, int):
        raise ValueError('non-numeric transport value')


def validate_contract(contract):
    assert contract['schema'] == SCHEMA and contract['prereg'] == PREREG
    assert contract['scope']['contains_pixels'] is False
    assert contract['scope']['contains_raw_GT'] is False
    assert contract['scope']['CNN_refit_from_public_numeric_verified'] is False
    expected = {('35', 'BEE24-35-head', 3), ('33', 'BEE24-33-head', 3),
                ('W', 'BEE24-10', 0), ('W', 'BEE24-15', 0),
                ('G', 'BEE24-24', 0), ('G', 'BEE24-28', 0)}
    actual = {(c['domain'], c['clip'], c['seed']) for c in contract['clips']}
    assert actual == expected and len(contract['clips']) == 6
    assert len(contract['folds']) == 4 and {f['heldout'] for f in contract['folds']} == {'35', '33', 'W', 'G'}
    for fold in contract['folds']:
        assert set(fold['source_domains']) == {'35', '33', 'W', 'G'} - {fold['heldout']}
        assert fold['training_seed'] == 208
    identifiers = [c['id'] for c in contract['components']]
    assert len(identifiers) == len(set(identifiers))
    for item in contract['components']:
        assert item['id'] == item['role'] + '|' + item['key']
        assert item['role'] in {'catalog', 'predictions', 'states', 'calibration',
                                'diagnostics', 'expected_summary', 'training_receipt',
                                'verification', 'frame_matching', 'diagnostic_rows'}
    public_numeric(contract)


def checked_components(contract, directory):
    """Explicit component paths only; future caller supplies finalized seals."""
    directory = Path(directory).resolve()
    for item in contract['components']:
        path = (directory / item['file']).resolve()
        assert path.is_relative_to(directory) and path.suffix == '.json'
        assert file_sha(path) == item['sha256']
        with path.open(encoding='utf-8') as stream:
            value = json.load(stream, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        public_numeric(value)
        yield item, value


def prepare(contract_path, directory, out, receipt_path, receipt_sha):
    """Only an explicit future finalized dispatch can write a new private package."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from tools.study202.stream202 import Writer
    assert re.fullmatch('[0-9a-f]{64}', receipt_sha) and file_sha(receipt_path) == receipt_sha
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    assert receipt['passed']
    contract = json.loads(Path(contract_path).read_text(encoding='utf-8'))
    validate_contract(contract)
    assert contract['verification_sha256'] == receipt_sha
    seals = receipt.get('input_seals', receipt.get('source_stage_seals'))
    assert seals is not None and contract['source_seals'] == seals
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    writer = Writer(out / 'shards')
    writer.write('contract', 'shared', contract)
    sources = {'contract': file_sha(contract_path), 'verification': receipt_sha}
    for item, value in checked_components(contract, directory):
        writer.write(item['role'], item['key'], value)
        sources[item['id']] = item['sha256']
        del value
    manifest = writer.finish(sources)
    manifest['schema'] = 'switch208-public-numeric-shards-v1'
    (out/'shards/manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n',
                                          encoding='utf-8', newline='\n')
    # Only explicit allowlisted numeric gzip objects ship; inflated scan files
    # remain private. Privacy review and equivalent reserialization precede use.
    names = sorted(['manifest.json'] + [r['file'] for r in manifest['shards']])
    with zipfile.ZipFile(out/'evidence.unreviewed.private.zip', 'x', compression=zipfile.ZIP_STORED) as archive:
        for name in names:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with (out/'shards'/name).open('rb') as source, archive.open(info, 'w') as target:
                shutil.copyfileobj(source, target, 65536)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--receipt-sha')
    parser.add_argument('--finalized-json-go', action='store_true')
    args = parser.parse_args()
    if not (args.finalized_json_go and args.contract and args.directory and args.out and args.receipt and args.receipt_sha):
        parser.error('Explicit finalized dispatch, normalized contract/directory, fresh output and pinned accepted receipt are required')
    prepare(args.contract, args.directory, args.out, args.receipt, args.receipt_sha)
