"""Private214 endpoint odd image and geometry features. No field/GT IO.

Caller enforces native-image whitelist, edge-distance and STRICT full2w*2h crop
eligibility before inference; BORDER_REPLICATE remains the specified sampler
behavior, not permission to include otherwise ineligible field endpoints.
Input image is ORIGINAL BGR uint8; output sampled patch is RGB native intensity.
"""
import cv2
import numpy as np


SCHEMA = 'center214.features.v1'
G_NAMES = ('log_width_over_height', 'native_score', 'time_role', 'vx', 'vy', 'log1p_nobs')
CHI_NAMES = tuple(f'odd_cell_y{y}_x{x}_{channel}' for y in range(2) for x in range(4) for channel in ('R', 'G', 'B'))


def _box(box_xywh):
    box = np.asarray(box_xywh, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
        raise ValueError('one finite positive xywh box required')
    return box


def sample_patch_rgb(image_bgr, box_xywh):
    """Specified16x16 pixel-center remap of centered2w*2h native image region."""
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8 or min(image_bgr.shape[:2]) < 1:
        raise ValueError('nonempty native BGR uint8 image[H,W,3] required')
    x, y, w, h = _box(box_xywh)
    cx, cy = x+w/2, y+h/2
    centers = (np.arange(16, dtype=np.float64)+.5)/16-.5
    map_x, map_y = np.meshgrid(cx+centers*2*w, cy+centers*2*h)
    patch = cv2.remap(image_bgr, map_x.astype(np.float32), map_y.astype(np.float32),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return patch[:, :, ::-1].copy()


def chi_from_patch(patch_rgb):
    """Per-channel standardize,4x4cell means, top-half odd component24 dims."""
    patch = np.asarray(patch_rgb, dtype=float)
    if patch.shape != (16, 16, 3) or not np.isfinite(patch).all():
        raise ValueError('finite16x16 RGB patch required')
    normalized = (patch-patch.mean(axis=(0, 1)))/np.maximum(patch.std(axis=(0, 1)), 1.)
    cells = normalized.reshape(4, 4, 4, 4, 3).mean(axis=(1, 3))
    odd = (cells-cells[::-1, ::-1])/2
    return odd[:2].reshape(24)


def image_chi(image_bgr, box_xywh):
    return chi_from_patch(sample_patch_rgb(image_bgr, box_xywh))


def endpoint_geometry(box_xywh, native_score, role, nobs, *, neighbor_center=None, frame_gap=None):
    """Own->same-run adjacent temporal observation, normalized by OWN w/h and|dt|.

    start uses chronological second row; end uses penultimate row. The neighbor
    is not another track or a second-nearest observation. Singleton vx=vy=role0.
    native_score is not calibrated/clipped; only vx/vy are clipped to±1.
    """
    box = _box(box_xywh)
    if role not in ('start', 'end', 'single') or isinstance(nobs, bool) or int(nobs) != nobs or nobs < 1 or not np.isfinite(native_score):
        raise ValueError('valid time role, positive integral nobs and finite native score required')
    own = box[:2]+box[2:]/2
    if role == 'single':
        if nobs != 1:
            raise ValueError('single role requires exactly one observation')
        velocity, time_role = np.zeros(2), 0.
    else:
        neighbor = np.asarray(neighbor_center, float)
        if nobs < 2 or neighbor.shape != (2,) or not np.isfinite(neighbor).all() or frame_gap is None or not np.isfinite(frame_gap) or frame_gap == 0:
            raise ValueError('non-single endpoint needs adjacent observation and nonzero native time gap')
        velocity = np.clip((neighbor-own)/box[2:]/abs(frame_gap), -1., 1.)
        time_role = 1. if role == 'start' else -1.
    return np.array([np.log(box[2]/box[3]), float(native_score), time_role,
                     float(velocity[0]), float(velocity[1]), np.log1p(nobs)])


def geometry_from_run(native_rows, endpoint_raw_index):
    """Convenience source adapter for one canonical run, original row indexing."""
    rows = np.asarray(native_rows, float)
    if rows.ndim != 2 or rows.shape[1] < 7 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError('one nonempty finite canonical run Nx7+ required')
    if isinstance(endpoint_raw_index, bool) or not isinstance(endpoint_raw_index, (int, np.integer)) or not 0 <= endpoint_raw_index < len(rows):
        raise ValueError('endpoint_raw_index must index supplied original run rows')
    if (rows[:, 0] < 1).any() or (rows[:, 0] != np.floor(rows[:, 0])).any() or len(np.unique(rows[:, 0])) != len(rows):
        raise ValueError('unique positive native integer dates required')
    order = np.argsort(rows[:, 0], kind='stable')
    if (np.diff(rows[order, 0]) > 25).any():
        raise ValueError('input must already be one original canonical gap run')
    index = int(endpoint_raw_index)
    if len(rows) == 1:
        return endpoint_geometry(rows[index, 2:6], rows[index, 6], 'single', 1)
    if index == order[0]:
        role, neighbor_index = 'start', order[1]
    elif index == order[-1]:
        role, neighbor_index = 'end', order[-2]
    else:
        raise ValueError('requested row is not a chronological endpoint')
    neighbor_center = rows[neighbor_index, 2:4]+rows[neighbor_index, 4:6]/2
    return endpoint_geometry(rows[index, 2:6], rows[index, 6], role, len(rows),
                             neighbor_center=neighbor_center, frame_gap=rows[neighbor_index, 0]-rows[index, 0])


def synthetic_assertions():
    cv2.setNumThreads(1)
    box = [3.5, 3.5, 8., 8.]
    flat = np.full((16, 16, 3), 97, np.uint8)
    np.testing.assert_array_equal(image_chi(flat, box), np.zeros(24))
    yy, xx = np.mgrid[:16, :16]
    bgr = np.stack((xx*8, yy*8, (xx+yy)*4), axis=-1).astype(np.uint8)
    patch = sample_patch_rgb(bgr, box)
    np.testing.assert_array_equal(patch, bgr[:, :, ::-1])
    chi = image_chi(bgr, box)
    assert chi.shape == (24,) and np.linalg.norm(chi) > 0
    np.testing.assert_allclose(chi_from_patch(patch[::-1, ::-1]), -chi, atol=1e-14)
    np.testing.assert_allclose(image_chi(bgr[::-1, ::-1].copy(), box), -chi, atol=1e-14)
    # Cross-image-border samples follow replicate exactly, not zero padding.
    cross = sample_patch_rgb(bgr, [-4.5, -4.5, 8., 8.])
    np.testing.assert_array_equal(cross[0, 0], bgr[0, 0, ::-1])
    assert np.isfinite(image_chi(bgr, [-4.5, -4.5, 8., 8.])).all()
    # Fractional shift reaches distinct interpolated intensities. Native8bit
    # cv2 remap rounds interpolation intensities, which this module preserves.
    fractional = sample_patch_rgb(bgr, [3.75, 3.75, 8., 8.])
    np.testing.assert_array_equal(fractional[0, 0], np.array([2, 2, 2], np.uint8))
    assert not np.array_equal(fractional, patch)
    rows = np.array([[3, 1, 4, 6, 2, 2, .7], [1, 1, 0, 0, 2, 2, .8]], float)
    start, end = geometry_from_run(rows, 1), geometry_from_run(rows, 0)
    np.testing.assert_allclose(start[2:5], [1., 1., 1.])
    np.testing.assert_allclose(end[2:5], [-1., -1., -1.])
    single = geometry_from_run(rows[:1], 0)
    np.testing.assert_array_equal(single[2:5], np.zeros(3))
    return dict(passed=True, schema=SCHEMA, chi_dimensions=len(chi), G_dimensions=len(start),
                checks='flat, odd180, border replicate, subpixel, temporal adjacent endpoint, singleton andvelocity clipping')


if __name__ == '__main__':
    print(synthetic_assertions())
