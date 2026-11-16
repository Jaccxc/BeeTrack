"""Temporal alignment diagnostic for the 25-fps, 30-offset BeeTrack count metric."""

import numpy as np


def _histogram(events, nframes):
    pairs = np.asarray(list(events))
    counts = np.zeros((nframes, 2), dtype=np.int64)
    if not pairs.size:
        return counts
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("events must be (frame, direction) pairs")
    frames, directions = pairs.T
    if not (np.isfinite(frames).all() and (frames == np.floor(frames)).all()
            and ((1 <= frames) & (frames <= nframes)).all()):
        raise ValueError("event frame outside the 1-based clip")
    if not np.isin(directions, [-1, 1]).all():
        raise ValueError("event direction must be +1 (in) or -1 (out)")
    np.add.at(counts, (frames.astype(int) - 1, (directions < 0).astype(int)), 1)
    return counts


def _window_edges(nframes):
    starts, ends = [], []
    for offset in range(0, 750, 25):
        edges = np.arange(-offset, nframes + 750, 750).clip(0, nframes)
        starts.extend(edges[:-1])
        ends.extend(edges[1:])
    return np.array(starts), np.array(ends)


def _counts(hist, starts, ends):
    cumulative = np.vstack(([0, 0], np.cumsum(hist, axis=0)))
    return cumulative[ends] - cumulative[starts]


def phase_summary(gt_events, pred_events, nframes, scale=(1, 1)):
    """Compare aligned timing to every nonzero whole-second circular prediction shift.

    `scale` is a fixed (in, out) count multiplier; use the same source-derived
    multiplier in all shifts. All errors/spreads are in percentage points of GT
    events. `fraction_le` is descriptive and includes ties.
    """
    if not isinstance(nframes, (int, np.integer)) or nframes <= 25:
        raise ValueError("clip must permit at least one nonzero 25-frame shift")
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (2,) or not np.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("scale requires two finite nonnegative factors")
    truth = _histogram(gt_events, nframes)
    if truth.sum() == 0:
        raise ValueError("normalized error requires at least one GT event")
    predicted = _histogram(pred_events, nframes)
    starts, ends = _window_edges(nframes)
    target = _counts(truth, starts, ends)
    denominator = 30 * truth.sum()
    assert target.sum() == denominator

    def error(shift):
        # Row 0 is frame 1, so rolling rows implements ((frame-1+shift) % nframes)+1.
        counts = _counts(np.roll(predicted, shift, axis=0), starts, ends)
        return float(100 * np.abs(counts * scale - target).sum() / denominator)

    original = error(0)
    shifted = np.array([error(shift) for shift in range(25, nframes, 25)])
    q10, median, q90 = np.quantile(shifted, [0.1, 0.5, 0.9])
    # Calibrated fractional counts can sum in a different order across shifts.
    tied = np.isclose(shifted, original, rtol=1e-12, atol=1e-12)
    return dict(original=original, shifted_median=float(median), gain_pp=float(median - original),
                shifted_q10=float(q10), shifted_q90=float(q90),
                shifted_sd_population=float(shifted.std()),
                shifted_sd_sample=float(shifted.std(ddof=1)) if shifted.size > 1 else None,
                fraction_le=float(np.mean((shifted < original) | tied)), n_shifts=len(shifted))


if __name__ == "__main__":
    n = 1575
    events = [(1, 1), (750, -1), (751, 1), (775, -1), (1575, -1), (1575, -1)]
    hist = _histogram(events, n)
    starts, ends = _window_edges(n)
    assert _counts(hist, starts, ends)[:3].tolist() == [[1, 1], [1, 1], [0, 2]]
    shifted = _counts(np.roll(hist, 25, axis=0), starts, ends)
    assert shifted[:3].tolist() == [[1, 2], [1, 2], [0, 0]]
    assert np.array_equal(shifted.sum(axis=0), 30 * hist.sum(axis=0))
    exact = phase_summary(events, events, n)
    assert exact["original"] == 0 and exact["n_shifts"] == 62 and exact["gain_pp"] > 0
    assert phase_summary(events, [], n)["fraction_le"] == 1
    scaled = phase_summary(events, events, n, scale=(2, 0.5))
    assert np.isclose(scaled["original"], 100 * (2 + 0.5 * 4) / 6)
