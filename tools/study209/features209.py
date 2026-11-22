"""Private209 image features. SOURCE ONLY until explicit field GO; no GT IO.

extract_pair(read_gray,native_boxes,boxes,frame,nframes,image_shape) accepts
1-based native dates, xywh boxes, read_gray(date,roi_xyxy)->uint8 ROI and
native_boxes(date)->all native xywh detections. Reader owns six-clip whitelist.
No frame reads on import. Main deployment support is TRUE-date validity only;
sham missingness must never filter primary G/F/X. Paired support is diagnostic.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
import cv2
import numpy as np


FEATURE_NAMES = ('flow', 'cross', 'board', 'within')
SCHEMA = 'common209.features.v1'


class BoundedROIReader:
    """Optional six-root-whitelisted native reader; cache ONLY bounded gray ROIs.

    path_for_frame(date) is injected, no directory scans or guessed filenames.
    Decode one transient grayscale frame, copy ROI, discard full frame. Single
    OpenCV thread. Caller must instantiate separately per clip to avoid date alias.
    """
    def __init__(self, path_for_frame, allowed_clip_roots, max_cached_rois=4):
        roots = tuple(Path(p).resolve() for p in allowed_clip_roots)
        if len(roots) != 6 or len(set(roots)) != 6 or max_cached_rois < 1:
            raise ValueError('exactly six distinct approved clip roots and positive cache bound required')
        self.path_for_frame, self.roots = path_for_frame, roots
        self.limit, self.cache = int(max_cached_rois), OrderedDict()

    def __call__(self, frame, roi):
        path = Path(self.path_for_frame(frame)).resolve()
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError('image outside approved six clip roots')
        key = (str(path), tuple(roi))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        cv2.setNumThreads(1)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('native image unavailable')
        x0, y0, x1, y1 = roi
        if not 0 <= x0 < x1 <= image.shape[1] or not 0 <= y0 < y1 <= image.shape[0]:
            raise ValueError('ROI outside native frame')
        gray = image[y0:y1, x0:x1].copy()
        del image
        self.cache[key] = gray
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return gray.copy()


def _boxes(values):
    a = np.asarray(values, float).reshape(-1, 4)
    if not np.isfinite(a).all() or (a[:, 2:] <= 0).any():
        raise ValueError('finite positive native xywh boxes required')
    return a


def _box_mask(boxes, roi):
    x0, y0, x1, y1 = roi
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = np.zeros(xx.shape, bool)
    for x, y, w, h in boxes:
        mask |= (xx >= x) & (xx < x+w) & (yy >= y) & (yy < y+h)
    return mask


def _masked_inputs(before, after, mask):
    """One fixed t-defined mask on BOTH dates, no other block's image evidence."""
    images = tuple(np.where(mask, gray, np.uint8(127)).astype(np.uint8) for gray in (before, after))
    support = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return images, support


def _track(before, after, mask, offset):
    # Corners AND forward/backward pyramidal LK receive ONLY these private
    # masked images. Unmasked native pixels never enter a block's LK pyramid.
    (before, after), support = _masked_inputs(before, after, mask)
    points = cv2.goodFeaturesToTrack(before, maxCorners=16, qualityLevel=.01,
                                    minDistance=3, blockSize=3, mask=support.astype(np.uint8)*255)
    if points is None:
        return dict(before=[], after=[], fb_error=[], status_forward=[], status_backward=[], keep=[], reasons=['no_corners'])
    nxt, forward, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, winSize=(15, 15), maxLevel=2)
    if nxt is None or forward is None:
        return dict(before=(points.reshape(-1, 2)+offset).tolist(), after=[], fb_error=[], status_forward=[], status_backward=[], keep=[], reasons=['forward_LK_failed'])
    finite_next = np.isfinite(nxt).all(axis=(1, 2))
    safe = nxt.copy(); safe[~finite_next] = points[~finite_next]
    back, backward, _ = cv2.calcOpticalFlowPyrLK(after, before, safe, None, winSize=(15, 15), maxLevel=2)
    if back is None or backward is None:
        return dict(before=(points.reshape(-1, 2)+offset).tolist(), after=(safe.reshape(-1, 2)+offset).tolist(), fb_error=[], status_forward=forward.ravel().astype(bool).tolist(), status_backward=[], keep=[], reasons=['backward_LK_failed'])
    fb = np.linalg.norm(back.reshape(-1, 2)-points.reshape(-1, 2), axis=1)
    p, q = points.reshape(-1, 2), nxt.reshape(-1, 2)
    in_roi = (q[:, 0] >= 0) & (q[:, 0] < before.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < before.shape[0])
    source_supported, target_supported = np.zeros(len(p), bool), np.zeros(len(p), bool)
    for index in range(len(p)):
        for coords, flags in ((p, source_supported), (q, target_supported)):
            if np.isfinite(coords[index]).all():
                x, y = np.rint(coords[index]).astype(int)
                if 0 <= x < support.shape[1] and 0 <= y < support.shape[0]:
                    flags[index] = support[y, x]
    keep = forward.ravel().astype(bool) & backward.ravel().astype(bool) & finite_next & np.isfinite(fb) & (fb <= 1.) & in_roi & source_supported & target_supported
    def finite_list(values):
        return [[float(v) if np.isfinite(v) else None for v in row] for row in values]
    return dict(before=finite_list(p+offset), after=finite_list(q+offset), fb_error=[float(v) if np.isfinite(v) else None for v in fb],
                status_forward=forward.ravel().astype(bool).tolist(), status_backward=backward.ravel().astype(bool).tolist(),
                source_supported=source_supported.tolist(), target_supported=target_supported.tolist(),
                masked_image_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in (before, after)],
                support_pixel_count=int(support.sum()), keep=keep.tolist(), reasons=[])


def _remove_collisions(traces):
    # Within each block both rounded dates must be unique. Across A/B, every
    # conflicting match is removed from BOTH sides, no order-dependent winner.
    for phase in ('before', 'after'):
        owners = {}
        for name, trace in traces.items():
            if not len(trace['keep']):
                continue
            for index in np.flatnonzero(trace['keep']):
                point = np.asarray(trace[phase][index])
                if np.isfinite(point).all():
                    owners.setdefault((name, *np.rint(point).astype(int)), []).append(index)
        for key, indices in owners.items():
            if len(indices) > 1:
                for index in indices:
                    traces[key[0]]['keep'][index] = False
                traces[key[0]]['reasons'].append(f'duplicate_rounded_{phase}_within_block')
    # Snapshot both phases BEFORE deletions to remove all conflicting points.
    conflicts = set()
    for phase in ('before', 'after'):
        maps = {}
        for name in ('a', 'b'):
            maps[name] = {}
            for index in np.flatnonzero(traces[name]['keep']):
                maps[name].setdefault(tuple(np.rint(traces[name][phase][index]).astype(int)), []).append(index)
        for pixel in maps['a'].keys() & maps['b'].keys():
            conflicts.update((name, index) for name in ('a', 'b') for index in maps[name][pixel])
    for name, index in conflicts:
        traces[name]['keep'][index] = False
        traces[name]['reasons'].append('cross_block_rounded_pixel_conflict')


def _fit(trace):
    keep = np.asarray(trace['keep'], bool)
    if keep.sum() < 4:
        return None, 'fewer_than_4_unique_FB_points'
    p, q = (np.asarray(trace[phase], float)[keep] for phase in ('before', 'after'))
    coverage = []
    for xy in (p, q):
        s = np.linalg.svd(xy-xy.mean(0), compute_uv=False)
        ratio = float(s[1]/s[0]) if s[0] > 0 else 0.
        coverage.append(ratio)
    trace['coverage_ratio'] = coverage
    if min(coverage) < .05:
        return None, 'insufficient_2D_coverage_ratio_below_.05'
    u, _, vt = np.linalg.svd((p-p.mean(0)).T @ (q-q.mean(0)))
    correction = np.eye(2); correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ correction @ u.T
    t = q.mean(0)-r @ p.mean(0)
    return dict(before=p, after=q, rotation=r, translation=t), None


def _errors(fit, target):
    return np.linalg.norm(target['before'] @ fit['rotation'].T+fit['translation']-target['after'], axis=1)


def _date_features(read_gray, native_boxes, boxes, date, nframes, image_shape):
    result = dict(valid=False, reason=None, frame=int(date), next_frame=int(date+1), features=None, feature_vector=None, traces={})
    if not 1 <= date < nframes:
        result['reason'] = 'no_next_native_date'; return result
    h, w = image_shape
    diag = float(np.linalg.norm(boxes[:, 2:], axis=1).mean())
    lo = boxes[:, :2].min(0)-.5*diag
    hi = (boxes[:, :2]+boxes[:, 2:]).max(0)+.5*diag
    roi = (max(0, int(np.floor(lo[0]))), max(0, int(np.floor(lo[1]))),
           min(w, int(np.ceil(hi[0]))), min(h, int(np.ceil(hi[1]))))
    result.update(roi=list(roi), mean_diagonal=diag)
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        result['reason'] = 'empty_ROI'; return result
    try:
        images = [read_gray(d, roi) for d in (date, date+1)]
    except (OSError, ValueError) as exc:
        result['reason'] = 'image_unavailable'; result['detail'] = str(exc); return result
    if any(im.dtype != np.uint8 or im.ndim != 2 or im.shape != (roi[3]-roi[1], roi[2]-roi[0]) for im in images):
        raise ValueError('reader must return native-scale uint8 ROI gray')
    am, bm = (_box_mask(boxes[i:i+1], roi) for i in (0, 1))
    # Local outer rectangle minus both fixed pair boxes and ALL native-date boxes.
    masks = dict(a=am & ~bm, b=bm & ~am,
                 background=~(am | bm | _box_mask(_boxes(native_boxes(date)), roi)))
    offset = np.array(roi[:2], float)
    traces = {name: _track(*images, mask, offset) for name, mask in masks.items()}
    _remove_collisions(traces)
    for name, trace in traces.items():
        trace['mask_pixel_count'] = int(masks[name].sum())
        trace['detected_count'] = len(trace['before'])
        trace['accepted_count'] = int(np.asarray(trace['keep'], bool).sum())
    result['traces'] = traces
    fits, failures = {}, []
    for name, trace in traces.items():
        fit, failure = _fit(trace)
        if failure:
            failures.append(f'{name}:{failure}')
        else:
            fits[name] = fit
            trace['rotation'] = fit['rotation'].tolist(); trace['translation'] = fit['translation'].tolist()
    if failures:
        result['reason'] = ';'.join(failures); return result
    a, b, background = (fits[name] for name in ('a', 'b', 'background'))
    errors = dict(a_to_b=_errors(a, b), b_to_a=_errors(b, a),
                  background_to_a=_errors(background, a), background_to_b=_errors(background, b),
                  a_to_a=_errors(a, a), b_to_b=_errors(b, b), background_to_background=_errors(background, background))
    result['errors_px'] = {name: values.tolist() for name, values in errors.items()}
    relative_support = [float(np.median(errors['background_to_'+name])) for name in ('a', 'b')]
    result['relative_motion_median_px'] = relative_support
    if min(relative_support) < .5:
        result['reason'] = 'relative_motion_support_below_.5px'; return result
    def rms(first, second):
        return float(np.sqrt(np.mean(np.concatenate((errors[first], errors[second]))**2))/diag)
    flow = [np.median(fit['after']-fit['before'], axis=0) for fit in (a, b)]
    result.update(valid=True, reason='valid', features=dict(cross=rms('a_to_b', 'b_to_a'),
                  board=rms('background_to_a', 'background_to_b'), within=rms('a_to_a', 'b_to_b'),
                  flow=float(np.linalg.norm(flow[0]-flow[1])/diag)))
    result['feature_vector'] = [result['features'][name] for name in FEATURE_NAMES]
    return result


def extract_pair(read_gray, native_boxes, boxes, frame, nframes, image_shape):
    """Frozen fixed A/B boxes on true and +/-25-date control images.

    Sham reads native_boxes(CONTROL date), never reuses true-date background mask.
    true.valid defines primary support. paired_valid only marks secondary frozen-X
    feature substitution diagnostics with unchanged original geometry/weights.
    All selected/rejected point coordinates, statuses and FB errors are retained.
    """
    cv2.setNumThreads(1)
    boxes = _boxes(boxes)
    if boxes.shape != (2, 4) or int(frame) != frame or int(nframes) != nframes or len(image_shape) != 2:
        raise ValueError('two fixed boxes, integral native dates, image_shape(H,W) required')
    frame, nframes = int(frame), int(nframes)
    true = _date_features(read_gray, native_boxes, boxes, frame, nframes, image_shape)
    control = frame+25 if frame+26 <= nframes else frame-25 if frame-25 >= 1 else None
    sham = (_date_features(read_gray, native_boxes, boxes, control, nframes, image_shape)
            if control is not None else dict(valid=False, reason='no_legal_25frame_control', frame=None, next_frame=None, features=None, feature_vector=None, traces={}))
    return dict(schema=SCHEMA, feature_names=list(FEATURE_NAMES), fixed_boxes_xywh=boxes.tolist(),
                true=true, sham=sham, main_valid=bool(true['valid']), sham_valid=bool(sham['valid']),
                paired_valid=bool(true['valid'] and sham['valid']))


def synthetic_mask_isolation_property():
    """Deferred tiny test: changing other-block pixels cannot alter this block.

    Checks all ordered A/B/background pairs, BOTH masked LK input dates bitwise,
    and the complete corner/LK trace. This is pixel-evidence isolation only;
    masks share declared geometry, and artificial127 boundaries can add stationary
    edges or cause exits/abstention. Those effects are NOT identity evidence.
    """
    cv2.setNumThreads(1)
    rng = np.random.default_rng(209)
    images = [rng.integers(0, 256, (64, 96), dtype=np.uint8) for _ in range(2)]
    yy, xx = np.mgrid[:64, :96]
    masks = dict(a=(xx >= 4) & (xx < 30) & (yy >= 4) & (yy < 60),
                 b=(xx >= 34) & (xx < 60) & (yy >= 4) & (yy < 60),
                 background=(xx >= 64) & (xx < 92) & (yy >= 4) & (yy < 60))
    checks = 0
    for protected in masks:
        original_inputs, original_support = _masked_inputs(*images, masks[protected])
        original_trace = _track(*images, masks[protected], np.array([0., 0.]))
        for changed in masks:
            if changed == protected:
                continue
            modified = [image.copy() for image in images]
            for image in modified:
                image[masks[changed]] = rng.integers(0, 256, int(masks[changed].sum()), dtype=np.uint8)
            updated_inputs, updated_support = _masked_inputs(*modified, masks[protected])
            assert np.array_equal(original_support, updated_support)
            assert all(np.array_equal(a, b) for a, b in zip(original_inputs, updated_inputs))
            assert original_trace == _track(*modified, masks[protected], np.array([0., 0.]))
            checks += 1
    return dict(passed=True, ordered_block_interventions=checks,
                scope='synthetic pixel isolation; not natural accuracy or identity evidence')
