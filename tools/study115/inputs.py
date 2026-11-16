"""GT-free three-detector MOT inputs for the E3 event exporter. `paths` only stats filenames."""
import argparse
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from beetrack.common import ROOT, RUNS


LINEAGE = {'source':'36fe1cc5470b9155a61bb8bed2f0f06242cccbbbda1b90842607976ce08e831e',
           'target':'011b06f9fb98325d0407beac86819fda5ea8058b6627934e34c06c403ae63375'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def group_paths(role, fold, part, first_seed, lineage_dir, runs_dir):
    if role not in LINEAGE or fold not in (33,35) or first_seed not in (0,3,6):
        raise ValueError('Expected a registered role, no33/no35 detector and group beginning at 0,3,6')
    if part not in (('head','tail') if role=='source' else ('head',)):
        raise ValueError('Expected a source head/tail or target head')
    hive = 68-fold if role=='source' else fold
    clip = f'BEE24-{hive}-{part}'
    seeds = list(range(first_seed,first_seed+3))
    paths = {s:(runs_dir/'trackers/dev'/f'q_lk_loho_no{fold}_s{s}'/'data'/f'{clip}.txt' if part=='head'
                else lineage_dir/'source/tail_mot'/f'{fold}_{s}_tail.txt') for s in seeds}
    meta = dict(role=role,fold=fold,hive=hive,part=part,first_seed=first_seed,seeds=seeds,clip=clip,
                nframes=3950 if part=='head' else 1000)
    return paths, meta


@lru_cache(maxsize=2)
def pinned_masks(path, checksum):
    raw = path.read_bytes()
    if sha(raw) != checksum:
        raise ValueError(f'Expected original prediction-only §108 masks: {path}')
    return json.loads(raw)


def read_group(role, fold, part, first_seed, lineage_dir, runs_dir, zones_path):
    """Read three whole native [frame,id,x,y,w,h,score] streams and verify §108 event descriptors."""
    from tools.study108.inputs108 import describe

    lineage_dir, runs_dir = Path(lineage_dir), Path(runs_dir)
    paths, meta = group_paths(role,fold,part,first_seed,lineage_dir,runs_dir)
    masks = pinned_masks(lineage_dir/f'{role}_masks.json',LINEAGE[role])
    if masks['role'] != role:
        raise ValueError('Wrong §108 prediction role')
    zones = Path(zones_path).read_bytes()
    polygon, zone_hash = json.loads(zones)[meta['clip']], sha(zones)
    streams, mot_hashes, prediction_hashes, mot_paths = {}, {}, {}, {}
    for seed, path in paths.items():
        key = f'{fold}_{seed}_{part}'
        ref = masks['predictions'][key]
        raw = path.read_bytes()
        digest = sha(raw)
        expected = (fold,seed,meta['clip'],meta['nframes'],f'loho_no{fold}_s{seed}','dev' if part=='head' else 'val')
        if (tuple(ref[k] for k in ('fold','seed','clip','nframes','tag','split')) != expected
                or ref['mot_sha256']!=digest or ref['sources']['beetrack/zones.json']!=zone_hash):
            raise ValueError(f'MOT, zones or case metadata differ from §108: {role}/{key}')
        if part=='head' and ref['sources'][path.relative_to(runs_dir).as_posix()]!=digest:
            raise ValueError(f'Source path differs from the §108 native MOT: {role}/{key}')
        rows = (np.loadtxt(io.BytesIO(raw), delimiter=',', usecols=range(7), ndmin=2)
                if raw.strip() else np.empty((0,7),dtype=float))
        if (not np.isfinite(rows).all() or not np.array_equal(rows[:,:2],np.rint(rows[:,:2]))
                or np.any(rows[:,0]<1) or np.any(rows[:,0]>meta['nframes']) or np.any(rows[:,4:6]<=0)):
            raise ValueError(f'Invalid native MOT: {role}/{key}')
        if describe(rows[:,:6],polygon) != ref['predictions']:
            raise ValueError(f'Crossing descriptors differ from §108: {role}/{key}')
        streams[seed] = rows
        mot_hashes[str(seed)], prediction_hashes[str(seed)] = digest, masks['prediction_sha256'][key]
        mot_paths[str(seed)] = (f'runs/{path.relative_to(runs_dir).as_posix()}' if part=='head'
                                else f'lineage/{path.relative_to(lineage_dir).as_posix()}')
    meta.update(masks_sha256=LINEAGE[role],zones_sha256=zone_hash,mot_sha256=mot_hashes,
                prediction_sha256=prediction_hashes,mot_paths=mot_paths)
    return streams, polygon, meta


def inventory(lineage_dir,runs_dir,zones_path):
    matrix, required = [], {zones_path}
    for role in ('source','target'):
        required.add(lineage_dir/f'{role}_masks.json')
        for fold in (33,35):
            for part in (('head','tail') if role=='source' else ('head',)):
                for group in (0,3,6):
                    paths, meta = group_paths(role,fold,part,group,lineage_dir,runs_dir)
                    required.update(paths.values())
                    matrix.append(dict(**meta,mot_present=sum(p.is_file() for p in paths.values())))
    missing = [str(p) for p in sorted(required) if not p.is_file()]
    return dict(groups=matrix,required_files=len(required),missing=missing)


def self_check():
    # Only the generated MOT, zones and explicit fake §108 masks are opened in this test.
    polygon = [[0,0],[10,0],[10,20],[0,20]]
    with TemporaryDirectory() as temp:
        root = Path(temp); lineage, runs, zones = root/'lineage', root/'runs', root/'zones.json'
        zones.write_text(json.dumps({f'BEE24-{h}-{p}':polygon for h in (33,35) for p in ('head','tail')}))
        zsha = sha(zones.read_bytes())
        mock = {}
        for role in LINEAGE:
            archive = dict(role=role,predictions={},prediction_sha256={})
            for fold in (33,35):
                for part in (('head','tail') if role=='source' else ('head',)):
                    for group in (0,3,6):
                        paths, meta = group_paths(role,fold,part,group,lineage,runs)
                        for seed,path in paths.items():
                            path.parent.mkdir(parents=True,exist_ok=True)
                            raw = (f'2,{seed+5},11,2,2,2,0.123,-1,-1,-1\n'
                                   f'3,{seed+5},8,2,2,2,0.456,-1,-1,-1\n').encode()
                            path.write_bytes(raw)
                            key = f'{fold}_{seed}_{part}'
                            sources = {'beetrack/zones.json':zsha}
                            if part=='head': sources[path.relative_to(runs).as_posix()] = sha(raw)
                            ref = dict(fold=fold,seed=seed,clip=meta['clip'],nframes=meta['nframes'],
                                       split='dev' if part=='head' else 'val',tag=f'loho_no{fold}_s{seed}',
                                       mot_sha256=sha(raw),sources=sources,predictions=[[seed+5,3,8,2,10,4]])
                            archive['predictions'][key] = ref
                            archive['prediction_sha256'][key] = sha(json.dumps(ref).encode())
            lineage.mkdir(exist_ok=True); raw=json.dumps(archive).encode()
            (lineage/f'{role}_masks.json').write_bytes(raw); mock[role]=sha(raw)
        report = inventory(lineage,runs,zones)
        assert len(report['groups'])==18 and report['required_files']==57 and not report['missing']
        with patch.dict(LINEAGE,mock):
            for spec in report['groups']:
                role,fold,part,first=(spec[k] for k in ('role','fold','part','first_seed'))
                streams,poly,meta=read_group(role,fold,part,first,lineage,runs,zones)
                assert list(streams)==spec['seeds'] and poly==polygon and meta['masks_sha256']==mock[role]
                assert all(np.array_equal(x[:,6],[.123,.456]) for x in streams.values())
            tail = lineage/'source/tail_mot/33_0_tail.txt'
            tail.write_bytes(tail.read_bytes()+b'\n')
            try:
                read_group('source',33,'tail',0,lineage,runs,zones)
            except ValueError as e:
                assert 'MOT, zones' in str(e)
            else:
                raise AssertionError('Altered MOT accepted')
            path = lineage/'source_masks.json'; wrong=json.loads(path.read_bytes())
            wrong['predictions']['33_0_head']['predictions'][0][1] += 1
            raw = json.dumps(wrong).encode(); path.write_bytes(raw)
            with patch.dict(LINEAGE,source=sha(raw)):
                try:
                    read_group('source',33,'head',0,lineage,runs,zones)
                except ValueError as e:
                    assert 'Crossing descriptors' in str(e)
                else:
                    raise AssertionError('Changed native descriptor accepted')
    print('Synthetic 18-group path, 7-column native MOT, descriptor and fingerprint checks passed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    commands.add_parser('self-check')
    paths=commands.add_parser('paths')
    paths.add_argument('--lineage-dir',type=Path,default=RUNS/'study108')
    paths.add_argument('--runs-dir',type=Path,default=RUNS)
    paths.add_argument('--zones-path',type=Path,default=ROOT/'beetrack/zones.json')
    args=parser.parse_args()
    if args.mode=='self-check':
        self_check()
    else:
        report=inventory(args.lineage_dir,args.runs_dir,args.zones_path)
        print(json.dumps(report,sort_keys=True))
        raise SystemExit(int(bool(report['missing'])))
