"""Synthetic-only G CPU precheck of mixed-training-recipe team composition.

Not a causal test of shared labels. Inputs are integer directional window bins;
each of three seeds contributes exactly one CS/VT arm. No file/data loader.
All errors use integer numerators scaled by eighteen, avoiding rounding ambiguity.
"""
from itertools import product

import numpy as np


def _integer(a, name):
    a = np.asarray(a)
    if (not np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_)
            or (a < 0).any()):
        raise ValueError(name + ' must contain nonnegative integer bins')
    # Object integers retain exact arithmetic for arbitrary count magnitudes.
    return a.astype(object)


def compare(cs, vt, gt):
    """cs/vt shapes (3,...,2); gt shape (...,2), same seed and bin ordering.

    Returns exact error numerators per bin and aggregate, seed/arm exposure,
    eight team errors, and each seed's direct arm errors. Divide aggregate error
    numerator by (18*sum(gt)) and multiply by 100 for pooled W (GT>0).
    """
    c, v, g = _integer(cs, 'CS'), _integer(vt, 'VT'), _integer(gt, 'GT')
    if c.shape != v.shape or c.ndim < 2 or c.shape[0] != 3 or c.shape[-1] != 2 or c.shape[1:] != g.shape:
        raise ValueError('Expected three paired seeds and matching directional GT bins')
    teams = []
    for bits in product((0, 1), repeat=3):
        members = np.stack([v[i] if bits[i] else c[i] for i in range(3)])
        # Team mean is sum/3. Eighteen times absolute error is 6*|sum-3g|.
        error18 = 6 * np.abs(members.sum(axis=0) - 3 * g)
        teams.append(dict(arms=bits, error18=error18, single_error18=18 * np.abs(members - g)))
    pure = [t for t in teams if len(set(t['arms'])) == 1]
    mixed = [t for t in teams if len(set(t['arms'])) != 1]
    # Exact integer sums first, division only after divisibility is verified.
    pure_sum = sum(t['error18'] for t in pure)
    mixed_sum = sum(t['error18'] for t in mixed)
    assert all(int(x) % 2 == 0 for x in pure_sum.flat)
    assert all(int(x) % 6 == 0 for x in mixed_sum.flat)
    pure18, mixed18 = pure_sum // 2, mixed_sum // 6
    a = np.abs(c.sum(axis=0) + v.sum(axis=0) - 6 * g)
    delta = v - c
    S = delta.sum(axis=0)
    # 6|m-g|=a, 6b0=|S|, 6bi=|S-2delta_i|. Scale mean by18.
    pure_formula18 = 3 * np.maximum(a, np.abs(S))
    mixed_formula18 = sum(np.maximum(a, np.abs(S - 2 * delta[i])) for i in range(3))
    assert np.array_equal(pure18, pure_formula18)
    assert np.array_equal(mixed18, mixed_formula18)
    exposure = lambda ts: [[sum(t['arms'][i] == arm for t in ts) for arm in (0, 1)] for i in range(3)]
    pure_exposure, mixed_exposure = exposure(pure), exposure(mixed)
    assert pure_exposure == [[1, 1]] * 3 and mixed_exposure == [[3, 3]] * 3
    # Each member's mean absolute error is exactly equal under both teamings.
    assert np.array_equal(3 * sum(t['single_error18'] for t in pure),
                          sum(t['single_error18'] for t in mixed))
    pure_single = sum(t['single_error18'].sum(axis=0) for t in pure)
    mixed_single = sum(t['single_error18'].sum(axis=0) for t in mixed)
    assert np.array_equal(3 * pure_single, mixed_single)
    return dict(teams=teams, pure_error18=pure18, mixed_error18=mixed18,
                pure_error18_total=int(pure18.sum()), mixed_error18_total=int(mixed18.sum()),
                difference_error18_total=int((mixed18 - pure18).sum()),
                gt_bin_sum=int(g.sum()), pure_exposure=pure_exposure, mixed_exposure=mixed_exposure,
                pure_single_error18_sum=pure_single, mixed_single_error18_sum=mixed_single,
                single_mean_equal=True,
                interpretation='Mixed training recipe team-composition effect; G CPU precheck only')


def self_test():
    # Opposing recipe errors: pure teams each miss by one, all mixed miss by 1/3.
    c = np.asarray([[[0, 0]]] * 3)
    v = np.asarray([[[2, 0]]] * 3)
    r = compare(c, v, np.asarray([[1, 0]]))
    assert (r['pure_error18_total'], r['mixed_error18_total']) == (18, 6)
    # A seed-paired pattern whose pure teams cancel, while mixed teams disturb it.
    c = np.asarray([[[0, 0]], [[3, 0]], [[0, 0]]])
    v = np.asarray([[[3, 0]], [[0, 0]], [[0, 0]]])
    r = compare(c, v, np.asarray([[1, 0]]))
    assert r['pure_error18_total'] == 0 and r['mixed_error18_total'] == 12
    rng = np.random.default_rng(190)
    for _ in range(100):
        c, v = rng.integers(0, 20, size=(2, 3, 30, 6, 2))
        g = rng.integers(0, 20, size=(30, 6, 2))
        compare(c, v, g)
        equal = compare(c, c, g)
        assert equal['difference_error18_total'] == 0
    compare(np.zeros((3, 1, 2), dtype=int), np.zeros((3, 1, 2), dtype=int), np.zeros((1, 2), dtype=int))
    try:
        compare(c.astype(float), v, g)
    except ValueError:
        pass
    else:
        raise AssertionError('Floating bins accepted')
    print('Synthetic eight-team enumeration, exposure, member-W equality and integer closed forms passed')


if __name__ == '__main__':
    self_test()
