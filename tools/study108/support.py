"""Find anchor crossings independently supported by either of two other detector seeds."""
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment


@lru_cache(maxsize=2)
def phase_lags(nframes):
    """99 ordered pairs of distinct cyclic offsets in seconds, fixed per clip length."""
    if nframes not in (1000, 3950):
        raise ValueError("Registered durations are 1000 and 3950 frames at 25 fps")
    seconds = nframes // 25
    available = range(6, seconds - 5)
    candidates = [(first, second) for first in available for second in available
                  if 6 <= abs(first-second) <= seconds-6]
    rng = np.random.Generator(np.random.PCG64(108))
    indices = rng.choice(len(candidates), size=99, replace=False)
    return tuple(candidates[i] for i in indices)


def _observations(records, nframes):
    rows = np.asarray(records, dtype=np.float64)
    if rows.shape == (0,):
        rows = rows.reshape(0, 6)
    if rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all():
        raise ValueError("Expected finite [id, signed frame, x1, y1, x2, y2] observations")
    ids, times = rows[:, :2].T.astype(np.int64)
    if not np.array_equal(rows[:, :2], np.column_stack((ids, times))) or len(np.unique(ids)) != len(ids):
        raise ValueError("Event IDs must be unique and IDs/frames integral")
    if np.any(times == 0) or np.any(np.abs(times) > nframes):
        raise ValueError("Event frame outside the clip")
    raw = rows[:, 2:] * 100
    coords = np.rint(raw).astype(np.int64)
    if not np.allclose(coords, raw, rtol=0, atol=1e-7) or np.any(coords[:, 2:] <= coords[:, :2]):
        raise ValueError("MOT boxes must have positive area and hundredth-pixel precision")
    order = np.argsort(ids, kind="stable")
    return ids[order], times[order], coords[order]


def _matches_by_lag(anchor, buddy, nframes, seconds):
    left, times, a = anchor
    right, other_times, b = buddy
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(0, high-low).prod(axis=2)
    areas = (a[:, 2:]-a[:, :2]).prod(axis=1)[:, None] + (b[:, 2:]-b[:, :2]).prod(axis=1)[None, :]
    # IoU >= 1/2 is equivalent to 3*intersection >= sum of the areas.
    spatial = (3*intersection >= areas) & (np.sign(times[:, None]) == np.sign(other_times[None, :]))
    anchor_times = np.abs(times[:, None])
    other_times = np.abs(other_times)
    matched, original_pairs = {}, []
    for lag in sorted({0, *seconds}):
        shifted = (other_times - 1 + 25*lag) % nframes + 1
        allowed = spatial & (np.abs(anchor_times-shifted[None, :]) <= 5)
        # Keep all rows and columns in resolved-ID order for the agreed SciPy tie handling.
        r, c = linear_sum_assignment(allowed, maximize=True)
        ok = allowed[r,c]
        r, c = r[ok], c[ok]
        matched[lag] = set(left[r].tolist())
        if lag == 0:
            original_pairs = np.column_stack((left[r], right[c])).tolist()
    return matched, original_pairs


def support_runs(anchor, other1, other2, nframes):
    """Return retained IDs, aligned buddy pairs, and counts for the 99 phase controls."""
    lags = phase_lags(nframes)
    first, second = zip(*lags)
    a, b, c = (_observations(rows, nframes) for rows in (anchor, other1, other2))
    seen1, pairs1 = _matches_by_lag(a, b, nframes, first)
    seen2, pairs2 = _matches_by_lag(a, c, nframes, second)
    return dict(retained=sorted(seen1[0] | seen2[0]), pairs=[pairs1,pairs2],
                phase_counts=[len(seen1[x] | seen2[y]) for x,y in lags],
                phase_lags=[list(pair) for pair in lags])


def _selfcheck():
    a = [[12,-30,0,0,2,1],[10,30,0,0,2,1],[11,30,0,0,2,1],[13,35,4,0,6,1]]
    b = [[50,35,0,0,1,1],[51,-30,0,0,1,1],[52,41,4,0,6,1]]
    c = [[100,35,4,0,6,1],[101,-35,0,0,.99,1]]
    result = support_runs(a,b,c,1000)
    assert result["pairs"] == [[[10,50],[12,51]], [[13,100]]]
    assert result["retained"] == [10,12,13] and result["phase_counts"] == [0]*99
    assert support_runs(a[::-1],b[::-1],c[::-1],1000) == result
    for n in (1000,3950):
        lags = phase_lags(n)
        s = n//25
        assert len(set(lags)) == 99
        for first,second in lags:
            assert all(min(d,s-d)>=6 for d in (first,second,abs(first-second)))
        assert support_runs([],[],[],n)["retained"] == []
        anchor = [[0,2,0,0,2,1]]
        lag = lags[0][0]
        # Under cyclic phase, a late buddy becomes frame 2 and appears only in the null.
        late = [[1,(1-25*lag)%n+1,0,0,2,1]]
        result = support_runs(anchor,late,[],n)
        assert result["retained"] == [] and result["phase_counts"][0] == 1
    print("Aligned and cyclic support checks passed for both durations.")


if __name__ == "__main__":
    _selfcheck()
