"""Adapt §126 portable or historical receipts to the raw §127 audit manifest.

No tracking is rerun. All ten registered MOTs and selected C row indices are hash
checked before a manifest is emitted. Run from a BeeTrack checkout with uv.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit127 import OFF

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def resolve(path, base):
    value = Path(path)
    return value if value.is_absolute() else base / value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source126', type=Path, nargs='+', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--gt33', type=Path, required=True)
    parser.add_argument('--gt35', type=Path, required=True)
    parser.add_argument('--polygon', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    zones = json.loads(args.polygon.read_text())
    entries, provenance = [], {}
    seen = set()
    for path in args.source126:
        receipt = json.loads(path.read_text())
        assert receipt['schema'] == 126 and receipt['complete']
        assert receipt['prereg_verified'] == 'c9ecd32a5632cd9420859c2718ffafad8dd38797'
        cells = [c for c in receipt['cells'] if
                 (c['family'] == 'gt' and c['arm'] in ('C', 'G')) or
                 (c['family'] == 'real' and c['arm'] == 'U' and c['seed'] in (0, 1, 2))]
        for hive in sorted({str(c['fold']) for c in cells}):
            assert hive in ('33', '35')
            seq = f'BEE24-{hive}-head'
            assert seq not in seen
            seen.add(seq)
            selected = [c for c in cells if str(c['fold']) == hive]
            assert len(selected) == 5 and all(c['seq'] == seq for c in selected)
            gtpath = getattr(args, f'gt{hive}')
            gt = np.loadtxt(gtpath, delimiter=',', ndmin=2)
            source = receipt['sources'][seq]
            assert sha(gtpath) == source['prepared_sha256']
            provenance_path = path.parent / f'{hive}_provenance.npz'
            with np.load(provenance_path, allow_pickle=False) as a:
                raw, prepared = a['raw'], a['prepared']
                saved_segments = a['segment']
            assert np.array_equal(prepared, gt)
            assert len(raw) and raw[:, 0].min() >= 1 and raw[:, 0].max() <= 3950
            tagged = np.c_[raw[:, :6], np.arange(len(raw))]
            _, first = np.unique(tagged[:, 0] * 1e6 + tagged[:, 1], return_index=True)
            duplicates = np.setdiff1d(np.arange(len(raw)), first)
            tagged[duplicates, 1] = tagged[:, 1].max() + 1 + np.arange(len(duplicates))
            tagged = OFF['merge_handovers'](tagged)
            rounded = tagged[:, :6].copy()
            rounded[:, 2:6] = np.array([[float(f'{x:.2f}') for x in row] for row in rounded[:, 2:6]])
            assert np.array_equal(rounded, gt[:, :6])
            assert np.array_equal(np.sort(tagged[:, 6].astype(int)), np.arange(len(raw)))
            official = OFF['by_track'](np.c_[gt[:, :6], tagged[:, 6]])
            segment = np.empty(len(raw), np.int64)
            segment[official[:, 6].astype(int)] = official[:, 1].astype(int)
            assert np.array_equal(segment, saved_segments)
            assert np.unique(official[:, :2], axis=0, return_counts=True)[1].max(initial=0) <= 1
            truth = OFF['zone_tracks'](official, zones[seq])
            oracle_path = resolve(source['oracle'], args.repo)
            assert sha(oracle_path) == source['oracle_sha256']
            oracle = np.load(oracle_path, allow_pickle=False)
            assert np.array_equal(oracle, np.c_[raw[:, 0],raw[:, 2:4],raw[:, 2:4]+raw[:, 4:6],np.ones(len(raw))])
            c = next(c for c in selected if c['arm'] == 'C')
            archive_path = resolve(c['archive'], path.parent)
            assert sha(archive_path) == c['archive_sha256']
            with np.load(archive_path, allow_pickle=False) as a:
                indices = a['selected_row0']
                assert np.array_equal(a['input'], oracle[indices])
            c_ids = sorted(set(map(int, segment[indices])))
            assert all(truth[g][0] is None for g in c_ids)
            expected = np.flatnonzero(np.isin(segment, [g for g in truth if truth[g][0] is None]))
            assert np.array_equal(indices, expected)
            provenance[seq] = dict(C_segment_ids=c_ids,source126_sha256=sha(path),
                                   provenance_sha256=sha(provenance_path),C_archive_sha256=sha(archive_path),
                                   prepared_sha256=sha(gtpath),oracle_sha256=sha(oracle_path))
            for cell in selected:
                for field in ('mot','archive'):
                    assert sha(resolve(cell[field], path.parent)) == cell[f'{field}_sha256']
                entries.append(dict(arm=cell['arm'],sequence=seq,seed=cell['seed'],gt=str(gtpath.resolve()),
                                    prediction=str(resolve(cell['mot'], path.parent).resolve()),
                                    polygon=str(args.polygon.resolve()),
                                    **({'c_segment_ids':c_ids} if cell['arm']=='C' else {})))
    assert seen == {'BEE24-33-head','BEE24-35-head'} and len(entries) == 10
    args.output.write_text(json.dumps(entries,indent=2), encoding='utf-8')
    args.output.with_suffix('.receipt.json').write_text(json.dumps(provenance,indent=2), encoding='utf-8')
    print('Verified 10 frozen MOTs and both C original-row provenance sets')

if __name__ == '__main__':
    main()
