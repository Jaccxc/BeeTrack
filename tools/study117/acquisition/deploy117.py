"""Run the §117 aligned P3 classifier on a video or ordered image directory, then export P0 rows."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def acquisition(directory):
    directory = Path(directory).resolve()
    spec = importlib.util.spec_from_file_location('acquisition117', directory/'run117.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'raster_mask') or not hasattr(module, 'Detector'):
        raise ValueError('Expected the §117 acquisition directory with the native detector')
    sys.path.insert(0, str(directory))
    return module


def source_frames(source, first=None, last=None, names=None):
    """Yield (source frame number, BGR image), preserving missing numbered frames and requested gaps."""
    import cv2
    import numpy as np

    if first is not None and first < 1 or last is not None and last < (first or 1):
        raise ValueError('Use positive, ordered one-based frame bounds')
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if all(p.stem.isdecimal() for p in files):
            dated = sorted((int(p.stem), p) for p in files)
            if dated and (dated[0][0] < 1 or len({n for n,_ in dated}) != len(dated)):
                raise ValueError('Numeric filenames must be unique positive frame numbers')
        else:
            dated = list(enumerate(files, 1))
        selected = set(names or ())
        if names and (len(selected) != len(names) or not selected <= {p.name for p in files}):
            raise ValueError('Requested image names must be distinct files in the source directory')
        for n, path in dated:
            if (first is None or n >= first) and (last is None or n <= last) and (not names or path.name in selected):
                image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Cannot decode image at source frame {n}')
                yield n, image
        return

    if names or not source.is_file():
        raise ValueError('A video file or image directory is required; --names selects directory frames')
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        n = 0
        while last is None or n < last:
            if not cap.grab():
                if last is not None and n < last:
                    raise ValueError(f'Video ended at frame {n} before the requested last frame')
                break
            n += 1
            if n >= (first or 1):
                ok, image = cap.retrieve()
                if not ok or image is None:
                    raise ValueError(f'Cannot decode video frame {n}')
                yield n, image
    finally:
        cap.release()


def native(detector, mask_fn, image):
    """One stock FP32 native forward; retain its decoded boxes, scores, P3 features and raw logits."""
    import torch

    tensor = detector.stock.preprocess([image])
    if tensor.dtype != torch.float32 or tensor.device.type != 'cpu' or tensor.shape[:2] != (1,3) or max(tensor.shape[2:]) != 640:
        raise ValueError('Expected a single 640 CPU FP32 input')
    before = detector.calls
    dense, raw = detector.stock.model(tensor)
    if detector.calls != before+1 or dense.shape[:2] != (1,5):
        raise ValueError('Expected one native single-class backbone/neck forward')
    p3 = raw['feats'][0].contiguous(memory_format=torch.channels_last)
    h,w = p3.shape[-2:]
    logits = raw['scores'][:, :, :h*w].reshape(1, 1, h, w)
    input_hw = tuple(tensor.shape[2:])
    valid, _ = mask_fn(image.shape[:2], input_hw, (h,w))
    return dense, p3, logits, valid, input_hw


def detection_rows(dense, native_logits, aligned_logits, input_hw, source_hw):
    import torch
    from ultralytics.utils import nms, ops

    n = native_logits.numel()
    if not torch.equal(aligned_logits, native_logits):
        # Preserve the stock sigmoid bits on unaltered P3 cells, and every P4/P5 value.
        altered = aligned_logits.flatten() != native_logits.flatten()
        scores = torch.where(altered, aligned_logits.sigmoid().flatten(), dense[0,4,:n])
        dense = dense.clone()
        dense[0,4,:n] = scores
    boxes = nms.non_max_suppression(dense, conf_thres=.05, iou_thres=.7, max_det=300, nc=1)[0]
    boxes[:, :4] = ops.scale_boxes(input_hw, boxes[:, :4], source_hw)
    return boxes[:, :5]


def detect(args, module, cpu):
    import numpy as np
    import torch
    from stream117 import TemporalP3
    from tools.study116.pilot_cpu import cpu_readback

    weights = args.weights
    if not weights.is_file() or weights.suffix.lower() != '.pt':
        raise ValueError('Supply a local single-class YOLO11s checkpoint (.pt)')
    model_sha = module.sha(weights)
    if args.weights_sha256 and args.weights_sha256.lower() != model_sha:
        raise ValueError('Weights differ from the requested SHA-256')
    if args.output.suffix.lower() != '.npy' or (args.stats and args.stats.resolve() == args.output.resolve()):
        raise ValueError('Use a .npy output and a separate optional stats file')
    stream = TemporalP3('A')
    detector = None
    rows, read, chosen, changed = [], 0, 0, 0
    source_digest = hashlib.sha256()
    origin, last = args.first, None
    with torch.inference_mode():
        for n,image in source_frames(args.source, args.first, args.last, args.names):
            if detector is None:
                detector = module.Detector(weights)
                cpu = cpu_readback(1)
            if origin is None:
                origin = n
            last = n
            read += 1
            source_digest.update(np.asarray([n,*image.shape], dtype='<i8').tobytes())
            source_digest.update(image.tobytes())
            dense, p3, logits, valid, input_hw = native(detector, module.raster_mask, image)
            stream_key = (model_sha, tuple(image.shape[:2]), input_hw)
            aligned, meta = stream.advance(stream_key, n, p3, logits, detector.head.cv3[0], valid)
            chosen += meta['current_ids'].numel()
            changed += int((logits != aligned['A']).sum())
            detections = detection_rows(dense, logits, aligned['A'], input_hw, image.shape[:2])
            rows.extend([n-origin+1, *row] for row in detections.tolist())
    nframes = (args.last or last) - origin + 1 if origin is not None and (args.last or last) is not None else 0
    buf = io.BytesIO()
    np.save(buf, np.asarray(rows, dtype=np.float64).reshape(-1,6), allow_pickle=False)
    module.write_once(args.output, buf.getvalue())
    report = dict(arm='A', source_first=origin, source_last=last, nframes=nframes, frames_read=read,
                  backbone_forwards=detector.calls if detector else 0, selected_cells=chosen, changed_logits=changed,
                  rows=len(rows), weights_sha256=model_sha, decoded_frames_sha256=source_digest.hexdigest(),
                  detections_sha256=module.sha(args.output), cpu=cpu, ultralytics='8.4.208')
    if args.stats:
        module.write_json(args.stats, report)
    return report


def self_check(module):
    """Use random single-class YOLO on synthetic frames plus a controlled P3 donor in its real head."""
    from collections import deque
    import numpy as np
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from decode117 import summarize
    from feature_arms import compute_arms
    from stream117 import TemporalP3

    torch.manual_seed(117)
    rng = np.random.default_rng(117)
    model = DetectionModel('yolo11s.yaml', nc=1, verbose=False).eval()
    detector = module.Detector(model)
    live = TemporalP3('A')
    reference = deque(maxlen=2)
    # Expose a short moving weak/strong pair within a real Detect forward. This makes the
    # synthetic test sensitive to a dropped or misregistered A correction with no trained data.
    class SingleLogit(torch.nn.Module):
        def forward(self, x):
            return 2*x[:,:1]-3.3
    detector.head.cv3[0] = SingleLogit()
    position = [None]
    def moving_feature(head, args):
        features = list(args[0])
        p3 = torch.zeros_like(features[0])
        y, x = p3.shape[-2]//2, p3.shape[-1]//2
        frame = position[0]
        feature = torch.arange(p3.shape[1], dtype=p3.dtype).remainder(7)-3
        feature[0] = 1.
        if frame in (1,3,7):
            p3[0,:,y,x+(frame != 1)] = feature * (2.5 if frame == 1 else .85)
        features[0] = p3
        return (features,)
    hook = detector.head.register_forward_pre_hook(moving_feature)
    affected, emitted = 0, 0
    try:
        with torch.inference_mode():
            for f in (1,2,3,6,7):
                position[0] = f
                image = rng.integers(0,256,size=(53,101,3),dtype=np.uint8)
                dense,p3,logits,mask,input_hw = native(detector,module.raster_mask,image)
                a,info = live.advance(('synthetic',image.shape[:2],input_hw), f,p3,logits,detector.head.cv3[0],mask)
                if reference and reference[-1][0] != f-1:
                    reference.clear()
                prev, past, boxes = reference[0][1:] if len(reference)==2 else (None,None,None)
                arms, ids = compute_arms(p3,prev,logits,past,detector.head.cv3[0],mask)
                assert torch.equal(a['A'],arms['A']) and torch.equal(info['current_ids'],ids['current_ids'])
                summary, source_boxes = summarize(dense,arms,ids,boxes,image.shape[:2],input_hw,past)
                reference.append((f,p3,logits,source_boxes))
                rows = detection_rows(dense,logits,a['A'],input_hw,image.shape[:2]).tolist()
                expected = [[*summary['boxes'][str(i)],s] for i,s in summary['arms']['A']]
                assert rows == expected
                if f == 3:
                    assert len(rows) == 1 and rows[0][4] >= .25 and summary['arms']['U'] != summary['arms']['A']
                    affected += len(info['current_ids'])
                else:
                    assert summary['arms']['A'] == summary['arms']['U'] and not info['current_ids'].numel()
                emitted += len(rows)
    finally:
        hook.remove()
        detector.hook.remove()
    assert detector.calls == 5 and affected == 1
    return dict(synthetic_frames=5, backbone_forwards=detector.calls, affected_cells=affected, output_rows=emitted,
                six_arm_A_exact=True, cold_and_gap_U_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    run = actions.add_parser('detect')
    run.add_argument('source',type=Path)
    run.add_argument('output',type=Path)
    run.add_argument('--weights',required=True,type=Path)
    run.add_argument('--weights-sha256')
    run.add_argument('--first',type=int,help='First source frame, 1-based (local output frame 1)')
    run.add_argument('--last',type=int,help='Last source frame, inclusive')
    run.add_argument('--names',nargs='+',help='Only these directory filenames, in source order')
    run.add_argument('--stats',type=Path)
    test = actions.add_parser('self-check')
    for p in (run,test):
        p.add_argument('--detector-dir',type=Path,default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    from tools.study116.pilot_cpu import configure_cpu
    cpu = configure_cpu(1)
    import ultralytics
    if ultralytics.__version__ != '8.4.208':
        raise ValueError('Use the registered Ultralytics version 8.4.208')
    module = acquisition(args.detector_dir)
    report = self_check(module) if args.action == 'self-check' else detect(args,module,cpu)
    print(json.dumps(report,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()
