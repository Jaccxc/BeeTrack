"""Freeze §113 background-probe predictions from four shared temporal median images."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from beetrack.common import DATA, ROOT, RUNS


NFRAMES = 3950
LINEAGE = {'source':'36fe1cc5470b9155a61bb8bed2f0f06242cccbbbda1b90842607976ce08e831e',
           'target':'011b06f9fb98325d0407beac86819fda5ea8058b6627934e34c06c403ae63375'}
PREDICT = dict(device='cpu', half=False, imgsz=640, conf=.05, iou=.7, max_det=300,
               augment=False, verbose=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wire(record):
    return (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def write_once(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        part = Path(temp)/'partial'
        part.write_bytes(raw)
        try:
            os.link(part, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError(f'Conflicting freeze: {path}')
    return sha(raw)


def case_paths(role, fold, seed):
    hive = 68-fold if role == 'source' else fold
    tag, clip = f'loho_no{fold}_s{seed}', f'BEE24-{hive}-head'
    return (hive, tag, clip, DATA/'train'/f'BEE24-{hive}'/'img1',
            RUNS/'detect'/tag/'weights/best.pt', RUNS/'trackers/dev'/f'q_lk_{tag}'/'data'/f'{clip}.txt')


def paths(args, frames):
    required = {ROOT/'beetrack/zones.json', args.lineage_dir/f'{args.role}_masks.json'}
    for fold in args.fold:
        for seed in args.seeds:
            _, _, _, folder, weights, mot = case_paths(args.role,fold,seed)
            required.update([weights,mot])
            required.update(folder/f'{f:06d}.jpg' for quarter in frames for f in quarter)
    missing = [str(path) for path in sorted(required) if not path.is_file()]
    print(json.dumps(dict(role=args.role,folds=args.fold,seeds=args.seeds,required=len(required),missing=missing)))
    return int(bool(missing))


def median_images(folder, frames, dest, clip, prereg):
    """Use BGR uint8 in native resolution. All seeds receive these same immutable pixel arrays."""
    entries, medians = [], []
    for quarter, numbers in enumerate(frames):
        if len(numbers) != 31 or len(set(numbers)) != 31:
            raise ValueError('Each quarter requires 31 unique samples')
        records = []
        for i, number in enumerate(numbers):
            raw = (folder/f'{number:06d}.jpg').read_bytes()
            image = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.dtype != np.uint8 or image.shape[2:] != (3,):
                raise ValueError(f'Cannot decode BGR uint8 frame {clip}/{number}')
            if i == 0:
                # Partition on the contiguous final axis, in place, to avoid a second 31-frame allocation.
                samples = np.empty((*image.shape,31), dtype=np.uint8)
            if image.shape != samples.shape[:-1] or (medians and image.shape != medians[0].shape):
                raise ValueError(f'Image dimensions changed in {clip}')
            samples[...,i] = image
            records.append([int(number),sha(raw)])
        samples.partition(15, axis=-1)
        median = np.ascontiguousarray(samples[...,15]); del samples
        buff = io.BytesIO(); np.save(buff,median,allow_pickle=False)
        name = f'quarter{quarter}.npy'
        fingerprint = write_once(dest/clip/name, buff.getvalue())
        entries.append(dict(quarter=quarter, samples=records, file=name, npy_sha256=fingerprint,
                            bgr_sha256=sha(median.tobytes())))
        medians.append(median)
    manifest = dict(study=113,prereg=prereg,clip=clip,nframes=NFRAMES,shape=list(medians[0].shape),
                    decode='opencv IMREAD_COLOR BGR uint8',opencv=cv2.__version__,quarters=entries)
    return medians, write_once(dest/clip/'manifest.json',wire(manifest))


def run_model(model, medians):
    outputs = []
    for median in medians:
        result, = model.predict(source=median, **PREDICT)
        boxes = result.boxes
        values = np.column_stack((boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()))
        if values.shape[1] != 5 or not np.isfinite(values).all():
            raise ValueError('Invalid static probe detections')
        # Keep original per-image detector order. The measurement assigns IDs before applying 0.25.
        outputs.append(values.tolist())
    return outputs


def freeze(args, frames):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from beetrack.evaluate import load_tracks
    from probe113 import measured_case

    if not re.fullmatch('[0-9a-f]{40}', args.prereg):
        raise ValueError('Expected full preregistration commit')
    # Set this before importing PyTorch or Ultralytics. The predictor also receives device=cpu explicitly.
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import torch_utils

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    # Ultralytics select_device resets the CPU thread count when setting up each predictor.
    torch_utils.NUM_THREADS = 2

    mask_data = (args.lineage_dir/f'{args.role}_masks.json').read_bytes()
    if sha(mask_data) != LINEAGE[args.role]:
        raise ValueError('Expected the byte-exact original §108 prediction-only masks')
    masks = json.loads(mask_data)
    zones_raw = (ROOT/'beetrack/zones.json').read_bytes()
    zones, zone_hash = json.loads(zones_raw), sha(zones_raw)
    for fold in args.fold:
        _, _, clip, folder, _, _ = case_paths(args.role,fold,args.seeds[0])
        medians, median_hash = median_images(folder,frames,args.outdir/'medians',clip,args.prereg)
        for seed in args.seeds:
            hive, tag, _, _, weights, mot_path = case_paths(args.role,fold,seed)
            key = f'{fold}_{seed}_head'
            ref = masks['predictions'][key]
            mot_bytes = mot_path.read_bytes()
            mot_hash = sha(mot_bytes)
            if (ref['clip'] != clip or ref['nframes'] != NFRAMES or ref['tag'] != tag
                    or ref['mot_sha256'] != mot_hash or ref['sources']['beetrack/zones.json'] != zone_hash
                    or ref['sources'][mot_path.relative_to(RUNS).as_posix()] != mot_hash):
                raise ValueError(f'Native MOT, polygon or case differs from §108: {args.role}/{key}')
            weight_hash = digest(weights)
            model = YOLO(str(weights))
            model.to('cpu'); model.model.float()
            with torch.inference_mode():
                output = run_model(model,medians)
            if digest(weights) != weight_hash:
                raise ValueError(f'Weights changed during inference: {weights}')
            measured = measured_case(load_tracks(io.BytesIO(mot_bytes)), zones[clip], output, NFRAMES)
            descriptors = sorted([e['id'], e['f_plus'] if e['direction']=='in' else -e['f_plus'],
                                  *(round(float(x),2) for x in e['post_xyxy'])] for e in measured['events'])
            if descriptors != ref['predictions']:
                raise ValueError(f'§113 event descriptors differ from §108 native events: {args.role}/{key}')
            hashes = dict(weights=weight_hash, mot=mot_hash, zones=zone_hash, medians=median_hash,
                          masks108=LINEAGE[args.role], predictions108=masks['prediction_sha256'][key])
            record = dict(study=113, prereg=args.prereg, role=args.role, fold=fold, hive=hive, clip=clip,
                          seed=seed, tag=tag, nframes=NFRAMES, sample_frames=frames, hashes=hashes,
                          weights_path=weights.relative_to(ROOT).as_posix(),mot_path=mot_path.relative_to(ROOT).as_posix(),
                          detector=dict(PREDICT,torch_threads=2,opencv_threads=1,torch=str(torch.__version__),
                                        ultralytics=ultralytics.__version__),
                          quarters=output, measurement=measured)
            path = args.outdir/args.role/f'{fold}_{seed}.json'
            print(f'{write_once(path,wire(record))} {path}',flush=True)


def inventory_inputs(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg):
        raise ValueError('Expected full preregistration commit')
    case_hashes, hives, median_hashes = {}, {}, {}
    for hive in (33,35):
        fold, rows = 68-hive, []
        clip = f'BEE24-{hive}-head'
        median_hash = digest(args.outdir/'medians'/clip/'manifest.json')
        median_hashes[clip] = median_hash
        for seed in range(9):
            key = f'{fold}_{seed}'
            raw = (args.outdir/'source'/f'{key}.json').read_bytes()
            case = json.loads(raw)
            if (tuple(case[k] for k in ('study','prereg','role','fold','hive','seed','clip','nframes')) !=
                    (113,args.prereg,'source',fold,hive,seed,clip,NFRAMES)
                    or case['hashes']['masks108'] != LINEAGE['source'] or case['hashes']['medians'] != median_hash):
                raise ValueError(f'Case provenance differs: {key}')
            measure = case['measurement']
            events = measure['events']
            touched = [event for event in events if event['selected_landmark'] is not None]
            counts = {d:dict(total=sum(e['direction']==d for e in events),
                             touched=sum(e['direction']==d for e in touched),
                             assessable=sum(e['direction']==d and e['reference_half'] is not None for e in events),
                             split_excluded=sum(e['direction']==d and e['reference_half'] is None for e in events))
                      for d in ('in','out')}
            if (len({e['id'] for e in events})!=len(events) or [e['id'] for e in touched]!=measure['touched_ids']
                    or any(measure['counts'][d][k]!=v for d,c in counts.items() for k,v in c.items())
                    or any(e['reference_half'] is None for e in touched)):
                raise ValueError(f'Event counts or cross-half eligibility differ: {key}')
            case_hashes[key] = sha(raw)
            rows.append(dict(seed=seed, counts=counts, m=len(touched)))
        hives[hive] = rows
    index = dict(study=113,prereg=args.prereg,role='source',cases=case_hashes,medians=median_hashes)
    return index, hives


def inventory(args):
    index, _ = inventory_inputs(args)
    path = args.outdir/'source_inventory.json'
    print(f'{write_once(path,wire(index))} {path}',flush=True)


def summary(args):
    frozen = (args.outdir/'source_inventory.json').read_bytes()
    if sha(frozen) != args.inventory_sha:
        raise ValueError('Source inventory SHA differs')
    index, observations = inventory_inputs(args)
    if json.loads(frozen) != index:
        raise ValueError('A source case or median changed after inventory freeze')
    hives = {}
    for hive, gt_count in ((33,184),(35,129)):
        rows = observations[hive]
        for r in rows:
            r['cap_pp'] = 100*r['m']/gt_count
        total_m = sum(r['m'] for r in rows)
        ordered_caps = sorted(r['cap_pp'] for r in rows)
        min_total_m = (2*9*gt_count+99)//100
        hives[str(hive)] = dict(fold=68-hive,gt_count=gt_count,seeds=rows,mean_m=total_m/9,
                                mean_m_by_direction={d:sum(r['counts'][d]['touched'] for r in rows)/9 for d in ('in','out')},
                                mean_cap_pp=100*total_m/(9*gt_count),median_cap_pp=ordered_caps[4],
                                range_cap_pp=[ordered_caps[0],ordered_caps[-1]],seeds_ge_2pp=sum(100*r['m']>=2*gt_count for r in rows),
                                total_m=total_m, required_m=min_total_m, eligible_for_followup=total_m>=min_total_m)
    report = dict(study=113,prereg=args.prereg,role='source',inventory_sha256=args.inventory_sha,
                  cases=index['cases'],hives=hives)
    path = args.outdir/'source_summary.json'
    print(f'{write_once(path,wire(report))} {path}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='command',required=True)
    preflight=modes.add_parser('paths',help='check filenames only; never open source data')
    measure=modes.add_parser('freeze',help='freeze CPU source predictions after preregistration')
    freeze_index=modes.add_parser('inventory',help='freeze fingerprints for all 18 sources before using GT totals')
    summarize=modes.add_parser('summary',help='compute source coverage bounds from all 18 frozen manifests')
    summarize.add_argument('--inventory-sha',required=True)
    for opt in (preflight,measure):
        opt.add_argument('--role',choices=('source','target') if opt is preflight else ('source',),default='source')
        opt.add_argument('--fold',nargs='+',choices=(33,35),type=int,default=[33,35])
        opt.add_argument('--seeds',nargs='+',choices=range(9),type=int,default=list(range(9)))
        opt.add_argument('--lineage-dir',type=Path,default=Path('D:/beetrack_runs/study108'))
    for command in (measure,freeze_index,summarize):
        command.add_argument('--prereg',required=True)
        command.add_argument('--outdir',type=Path,default=Path('D:/beetrack_runs/study113'))
    args=parser.parse_args()
    if args.command == 'inventory':
        return inventory(args)
    if args.command == 'summary':
        return summary(args)
    if args.command == 'freeze':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from probe113 import quarter_frames
    frames = [[int(f) for f in q] for q in quarter_frames(NFRAMES)]
    return paths(args,frames) if args.command=='paths' else freeze(args,frames)


if __name__=='__main__':
    raise SystemExit(main())
