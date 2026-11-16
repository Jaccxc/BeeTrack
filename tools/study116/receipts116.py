"""Read the 24 sealed §116 inference arms and replay their shared-forward receipts."""
from collections import defaultdict
import hashlib
import json
from math import fsum, isfinite
from pathlib import Path
import re

from verify_dose import close, tensor_gains, verify_dose


PREREG = 'c8fabf6c55c9737086b02050fabb3360bb5f1db2'
GEOMETRY_SHA = 'f5c9e954977ee1cab4eb04359c9c30b4a2940ae6e1c2753fc041968e1bc02a78'
ARMS = 'ULCH'
CLIPS = {'BEE24-33-tail': 35, 'BEE24-35-tail': 33}
SOURCES = ('requests_file_sha256', 'weights_sha256', 'zones_sha256', 'code_sha256', 'prereg')
SHAPES = ('source_hw', 'tensor_bchw', 'forward_count', 'letterbox')
TIMES = ('warp', 'preprocess', 'forward', 'native_nms', 'wrapped_nms', 'total',
         'decode_shared', 'total_with_shared_decode')
CODE_NAMES = {'run_pilot.py', 'entrance_scale.py', 'inspector.py', 'pilot_cpu.py',
              'warp_predict.py', 'predictor.py', 'predict.py', 'nms.py', 'ops.py'}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False) + '\n').encode('utf8')
    return hashlib.sha256(raw).hexdigest()


def sha256(value):
    check(isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value), 'Invalid SHA-256')


def read_json(path, expected):
    raw = path.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == expected, f'File fingerprint mismatch: {path.name}')
    return json.loads(raw)


def cpu_limits(cpu):
    keys = {'affinity', 'torch', 'interop', 'opencv', 'ultra_threads',
            'ultra_select_device_threads', 'native_pools'}
    check(set(cpu) == keys, 'Unexpected CPU receipt fields')
    cores = cpu['affinity']
    n = cpu['torch']
    check(cores == sorted(set(cores)) and 1 <= len(cores) <= n <= 2
          and all(type(v) is int and v >= 0 for v in cores), 'Invalid physical CPU affinity')
    check(cpu['interop'] == cpu['opencv'] == 1 and n == cpu['ultra_threads']
          == cpu['ultra_select_device_threads'], 'Invalid CPU threading readback')
    for name, count in cpu['native_pools'].items():
        check(re.fullmatch('[A-Za-z0-9_.-]+', name) and type(count) is int and 1 <= count <= n,
              'Invalid native thread-pool readback')


def file_keys(clip, seed, arm):
    suffix = f'{clip}/{CLIPS[clip]}_{seed}_{arm}.json'
    return [f'{directory}/{suffix}' for directory in ('predictions', 'wrapped_predictions')]


def check_done(done):
    check(set(done) == {'study', 'prereg', 'files', 'code_sha256'} and done['study'] == 116
          and done['prereg'] == PREREG, 'Invalid global inference seal')
    keys = {name for clip in CLIPS for seed in range(3) for arm in ARMS
            for name in file_keys(clip, seed, arm)}
    check(set(done['files']) == keys and set(done['code_sha256']) == CODE_NAMES,
          'Missing or unexpected sealed run or detector module')
    for value in [*done['files'].values(), *done['code_sha256'].values()]:
        sha256(value)


def same_or_set(mapping, key, value):
    check(mapping.setdefault(key, value) == value, f'Shared receipt field changed: {key}')


def collect_receipts(root, done):
    """Read only globally sealed predictions, receipts, and the public camera geometry."""
    root = Path(root)
    check_done(done)
    sealed = read_json(root/'detector_done.json', digest(done))
    check(sealed == done, 'Global inference seal changed during packing')
    data = {'geometry': read_json(root/'geometry.json', GEOMETRY_SHA), 'detector': done,
            'detector_sha256': digest(done), 'geometry_sha256': GEOMETRY_SHA, 'clips': {}}
    for clip, fold in CLIPS.items():
        camera = data['clips'][clip] = {'images': {}, 'seeds': {}}
        for seed in range(3):
            cell = camera['seeds'][str(seed)] = {'arms': {}}
            for arm in ARMS:
                paths = file_keys(clip, seed, arm)
                pred, stock = [read_json(root/key, done['files'][key]) for key in paths]
                run = pred['run']
                check(run == dict(stock['run'], nms_space='original') and
                      stock['run']['nms_space'] == 'warped', 'Differing sources between paired NMS')
                checked = {'fold': fold, 'seed': seed, 'dose': arm, 'imgsz': 960 if arm == 'H' else 640,
                           'requested_gain': 1.5 if arm in 'LC' else 1., 'torch_dtype': 'float32',
                           'rect': True, 'iou': .7, 'conf': .05, 'max_det': 300,
                           'original_start': 4001, 'original_end': 5000, 'prereg': PREREG,
                           'code_sha256': done['code_sha256'], 'zones_sha256': data['geometry']['zones_sha256']}
                check(all(run[k] == v for k,v in checked.items()), 'Registered inference settings differ')
                same_or_set(data, 'cpu', run['cpu'])
                same_or_set(cell, 'configuration', {k: run[k] for k in SOURCES})
                frame_sha = run['frame_receipts_sha256']
                frames = pred['requested_frames']
                same_or_set(camera, 'frames', frames)
                request = {'clip': clip, 'nframes': 1000, 'frames': frames}
                check(set(frame_sha) == {str(f) for f in frames} and digest(request) == run['requests_file_sha256'],
                      'Requested frame sequence differs from sealed list')
                for result in (pred, stock):
                    check(result['clip'] == clip and result['status'] == 'available' and
                          result['requested_frames'] == frames and result['requests_sha256'] == digest(request)
                          and set(result['predictions']) == set(frame_sha) and
                          digest(result['predictions']) == result['predictions_sha256'], 'Frozen rows do not match request')
                stream = cell['arms'][arm] = {'frames': {}, 'predictions_sha256': pred['predictions_sha256'],
                                              'wrapped_sha256': stock['predictions_sha256']}
                for f in frames:
                    key = str(f)
                    fname = root/'receipts'/clip/f'{fold}_{seed}_{arm}'/f'{f:04}.json'
                    receipt = read_json(fname, frame_sha[key])
                    same_or_set(camera['images'], key, receipt['image_sha256'])
                    same_or_set(stream, 'shape', {k: receipt[k] for k in SHAPES})
                    first, second = pred['predictions'][key], stock['predictions'][key]
                    check(first == receipt['predictions'] and second == receipt['wrapped'], 'Paired output rows changed')
                    check(arm not in 'UH' or first == second, 'Unwarped original differs from package NMS')
                    stream['frames'][key] = {'receipt_sha256': frame_sha[key], 'timing_ms': receipt['timing_ms']}
                    check_rows(data, clip, seed, arm, f, first, second)
    return data


def check_rows(data, clip, seed, arm, frame, original, wrapped):
    """Rebuild the entire private receipt byte digest using the two published FP32 row streams."""
    camera = data['clips'][clip]
    cell = camera['seeds'][str(seed)]
    stream = cell['arms'][arm]
    f = str(frame)
    measured = stream['frames'][f]
    raw = dict(clip=clip, fold=CLIPS[clip], seed=int(seed), dose=arm, frame=int(frame),
               original_frame=4000+int(frame), image_sha256=camera['images'][f],
               configuration_sha256=digest(cell['configuration']), timing_ms=measured['timing_ms'],
               predictions=original, wrapped=wrapped, **stream['shape'])
    check(digest(raw) == measured['receipt_sha256'], 'Joint NMS receipt does not reproduce its sealed SHA-256')


def check_manifests(data, clip, seed, arm, original, wrapped):
    """Reproduce each sealed detector manifest from {native frame string: detector rows}."""
    camera = data['clips'][clip]
    cell = camera['seeds'][str(seed)]
    stream = cell['arms'][arm]
    files = data['detector']['files']
    registered = dict(fold=CLIPS[clip], seed=int(seed), dose=arm, imgsz=960 if arm == 'H' else 640,
                      requested_gain=1.5 if arm in 'LC' else 1., cpu=data['cpu'],
                      frame_receipts_sha256={f: row['receipt_sha256'] for f,row in stream['frames'].items()},
                      torch_dtype='float32', rect=True, iou=.7, conf=.05, max_det=300, status='available',
                      original_start=4001, original_end=5000, **cell['configuration'])
    expected = [stream['predictions_sha256'], stream['wrapped_sha256']]
    for space, paths, rows, frozen in zip(('original', 'warped'), file_keys(clip, seed, arm),
                                          (original, wrapped), expected):
        check(set(rows) == set(stream['frames']) and digest(rows) == frozen, 'Sealed prediction rows changed')
        payload = dict(format='warp-pilot-predictions-v1', clip=clip, nframes=1000,
                       requested_frames=camera['frames'], run=dict(registered, nms_space=space),
                       status='available', predictions=rows, requests_sha256=registered['requests_file_sha256'],
                       predictions_sha256=frozen)
        check(digest(payload) == files[paths], 'Detector manifest differs from global seal')


def replay_receipts(data):
    """Recompute the three geometry bands, check physical receipts and return per-arm mean timings."""
    done = data['detector']
    check_done(done)
    cpu_limits(data['cpu'])
    geo = data['geometry']
    check(digest(done) == data['detector_sha256'] and digest(geo) == data['geometry_sha256'] == GEOMETRY_SHA,
          'Changed camera calibration or inference manifest')
    check(set(geo['cases']) == set(data['clips']) == set(CLIPS), 'Unexpected clip population')
    dose, cases, total = {}, {}, 0
    arm_totals = {arm: defaultdict(list) for arm in ARMS}
    arm_counts = {arm: 0 for arm in ARMS}
    for clip, camera in data['clips'].items():
        cal = geo['cases'][clip]
        verify_dose(cal)
        sha256(cal['seqinfo_sha256'])
        dose[clip] = dict(g=cal['L']['gain'], delta_j=cal['dose_difference'],
                          n_points=len(cal['samples_xy_weight']),
                          main_rect=cal['L']['rect'], control_rect=cal['C']['rect'],
                          area_fraction=cal['L']['area_fraction'], overlap_px2=cal['rect_overlap_px2'],
                          rect_iou=cal['rect_iou'], L_xyj=cal['L']['mean_xyj'],
                          C_xyj=cal['C']['mean_xyj'], U_xyj=cal['U_xyj'], H_xyj=cal['H_xyj'])
        frames = camera['frames']
        keys = {str(f) for f in frames}
        check(frames == sorted(set(frames)) and all(type(f) is int and 1 <= f <= 1000 for f in frames)
              and 0 < len(frames) <= 32 and set(camera['images']) == keys, 'Invalid requested frames')
        request_sha = digest(dict(clip=clip, frames=frames, nframes=1000))
        for image_sha in camera['images'].values():
            sha256(image_sha)
        check(set(camera['seeds']) == {'0','1','2'}, 'Missing seed realization')
        by_seed = cases[clip] = {}
        for seed, cell in camera['seeds'].items():
            config = cell['configuration']
            check(set(config) == set(SOURCES) and config['prereg'] == PREREG and
                  config['code_sha256'] == done['code_sha256'] and config['zones_sha256'] == geo['zones_sha256']
                  and config['requests_file_sha256'] == request_sha, 'Unpinned inference sources')
            sha256(config['weights_sha256'])
            arms = cell['arms']
            h_over_u = tensor_gains(cal['size'], {arm: row['shape'] for arm,row in arms.items()})
            close(h_over_u, cal['H_xyj'], 'Actual H/U letterbox gain disagrees with geometry')
            by_arm = by_seed[seed] = {}
            for arm, stream in arms.items():
                values = defaultdict(list)
                check(set(stream['frames']) == keys, 'Missing frame receipt')
                sha256(stream['predictions_sha256'])
                sha256(stream['wrapped_sha256'])
                check(arm not in 'UH' or stream['predictions_sha256'] == stream['wrapped_sha256'],
                      'Unwarped NMS streams differ')
                for row in stream['frames'].values():
                    sha256(row['receipt_sha256'])
                    times = row['timing_ms']
                    check(set(times) == set(TIMES) and all(isfinite(v) and v >= 0 for v in times.values()),
                          'Invalid per-stage CPU timing')
                    close(times['total_with_shared_decode'], times['total']+times['decode_shared'], 'Shared decode timing')
                    for key in TIMES:
                        values[key].append(times[key])
                count = len(frames)*stream['shape']['forward_count']
                total += count
                arm_counts[arm] += count
                for k,v in values.items():
                    arm_totals[arm][k].append(fsum(v))
                by_arm[arm] = dict(forwards=count, tensor_bchw=stream['shape']['tensor_bchw'],
                                   mean_ms={k: fsum(v)/len(v) for k,v in values.items()})
    by_arm = {arm: dict(forwards=arm_counts[arm], mean_ms={k: fsum(v)/arm_counts[arm]
                       for k,v in arm_totals[arm].items()}) for arm in ARMS}
    return dose, dict(cases=cases, by_arm=by_arm, forwards=total, cpu=data['cpu'])
