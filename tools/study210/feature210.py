"""Private portable210 pure prediction geometry features; no GT or frame IO.

Inputs: each native run is Nx6+ MOT [date,id,x,y,w,h,...], with unique native
dates. Both whole runs must have NO common date. Centers use native xywh, no
rounding, no coordinate fitting at test points. Brackets interpolate ONLY
between adjacent observed outer rows, each side1..3 actual frames. Singletons
nested between two outer observations are legal. All offsets have sign B-A.

Common scale is median diagonal of UNIQUE supported (track,date) original boxes.
Full support requires both orientations anywhere, midpoint=floor((earliest+
latest supported raw dates)/2), >=2 fully-before and >=2 fully-after brackets.
Cross-cut brackets are discarded. Prefix coordinate median estimates b, never
suffix; original rows are disjoint between prefix/suffix by construction.

G6 and X2 are fixed below. Unsupported X/Xminus and G suffix-RMS are zero with
full_support=False explicitly reported. Xminus changes ONLY b to-b. No outputs
claim same identity: parallel separate bees have identical observations, and
acceleration can create interpolation residual even for the same physical bee.
"""
import numpy as np


SCHEMA = 'interleave210.features.v1'
G_NAMES = ('min_offset_norm', 'median_offset_norm', 'log1p_brackets',
           'orientation_balance', 'full_support', 'suffix_zero_rms')
X_NAMES = ('prefix_offset_norm', 'suffix_zero_minus_offset_rms')


def _run(rows, track):
    a = np.asarray(rows, float)
    if a.ndim != 2 or a.shape[1] < 6 or not np.isfinite(a).all():
        raise ValueError('finite Nx6+ native MOT rows required')
    if ((a[:, 0] < 1).any() or (a[:, 0] != np.floor(a[:, 0])).any()
            or len(np.unique(a[:, 0])) != len(a) or (a[:, 4:6] <= 0).any()):
        raise ValueError('unique positive integral dates and positive xywh extents required')
    order = np.argsort(a[:, 0], kind='stable')
    a = a[order]
    return dict(rows=a, dates=a[:, 0].astype(np.int64),
                center=a[:, 2:4]+a[:, 4:6]/2, diagonal=np.linalg.norm(a[:, 4:6], axis=1),
                keys=[(track, int(date)) for date in a[:, 0]])


def pair_features(track_a, track_b):
    """Return JSON-safe numeric features PLUS full source/interpolation evidence.

    Caller identifies canonical track runs/pair ID; this function does not group
    field tracks, choose candidate pairs or inspect identities. Invalid pair has
    G/X=None (not a fitted zero candidate). A bracket-valid but unsupported pair
    retains G with full_support=0 and zero X controls.
    """
    a, b = _run(track_a, 'A'), _run(track_b, 'B')
    result = dict(schema=SCHEMA, valid=False, reason=None, G_names=list(G_NAMES),
                  X_names=list(X_NAMES), G=None, X=None, Xminus=None,
                  full_support=False, common_scale=None, brackets=[],
                  prefix_indices=[], suffix_indices=[], cutoff_frame=None,
                  prefix_offset_b_minus_a=None, suffix_zero_rms=None,
                  suffix_residual_rms=None, suffix_wrong_sign_rms=None)
    if not len(a['dates']) or not len(b['dates']):
        result['reason'] = 'empty_run'; return result
    if np.intersect1d(a['dates'], b['dates']).size:
        result['reason'] = 'whole_pair_has_common_observation_date'; return result
    if max(a['dates'][0], b['dates'][0]) >= min(a['dates'][-1], b['dates'][-1]):
        # A singleton can be nested INSIDE the other run's span. Its own span
        # has zero length, so test the closed interval overlap separately.
        if max(a['dates'][0], b['dates'][0]) > min(a['dates'][-1], b['dates'][-1]):
            result['reason'] = 'nonoverlapping_run_spans'; return result
    brackets = []
    for middle, outside, name in ((b, a, 'B'), (a, b, 'A')):
        for index, date in enumerate(middle['dates']):
            right = int(np.searchsorted(outside['dates'], date))
            if right == 0 or right == len(outside['dates']):
                continue
            left = right-1
            t0, t1 = int(outside['dates'][left]), int(outside['dates'][right])
            if not (1 <= date-t0 <= 3 and 1 <= t1-date <= 3):
                continue
            weight = float((date-t0)/(t1-t0))
            interpolated = (1-weight)*outside['center'][left]+weight*outside['center'][right]
            delta = middle['center'][index]-interpolated if name == 'B' else interpolated-middle['center'][index]
            keys = (outside['keys'][left], middle['keys'][index], outside['keys'][right])
            brackets.append(dict(orientation='ABA' if name == 'B' else 'BAB',
                                 dates=[t0, int(date), t1], row_keys=[list(k) for k in keys],
                                 interpolation_weight=weight, interpolated_center=interpolated.tolist(),
                                 observed_middle_center=middle['center'][index].tolist(),
                                 offset_b_minus_a=delta.tolist()))
    brackets.sort(key=lambda r: (r['dates'][1], r['orientation']))
    result['brackets'] = brackets
    if not brackets:
        result['reason'] = 'no_adjacent_narrow_bracket'; return result
    supported = {tuple(key) for bracket in brackets for key in bracket['row_keys']}
    diag_lookup = {key: float(diagonal) for run in (a, b) for key, diagonal in zip(run['keys'], run['diagonal'])}
    scale = float(np.median([diag_lookup[key] for key in sorted(supported)]))
    all_dates = [key[1] for key in supported]
    cutoff = (min(all_dates)+max(all_dates))//2
    prefix = [i for i, bracket in enumerate(brackets) if max(bracket['dates']) <= cutoff]
    suffix = [i for i, bracket in enumerate(brackets) if min(bracket['dates']) > cutoff]
    aba = sum(r['orientation'] == 'ABA' for r in brackets)
    bab = len(brackets)-aba
    full = aba > 0 and bab > 0 and len(prefix) >= 2 and len(suffix) >= 2
    offsets = np.asarray([bracket['offset_b_minus_a'] for bracket in brackets])
    magnitude = np.linalg.norm(offsets, axis=1)/scale
    g = [float(magnitude.min()), float(np.median(magnitude)), float(np.log1p(len(brackets))),
         float(min(aba, bab)/len(brackets)), float(full), 0.]
    x, wrong = [0., 0.], [0., 0.]
    if full:
        prefix_keys = {tuple(key) for i in prefix for key in brackets[i]['row_keys']}
        suffix_keys = {tuple(key) for i in suffix for key in brackets[i]['row_keys']}
        assert not prefix_keys & suffix_keys
        estimate = np.median(offsets[prefix], axis=0)
        actual = offsets[suffix]
        def rms(values):
            return float(np.sqrt(np.mean(np.sum(values**2, axis=1))))
        zero, residual, wrong_sign = rms(actual), rms(actual-estimate), rms(actual+estimate)
        g[-1] = zero/scale
        x = [float(np.linalg.norm(estimate)/scale), (zero-residual)/scale]
        wrong = [x[0], (zero-wrong_sign)/scale]
        result.update(prefix_offset_b_minus_a=estimate.tolist(), suffix_zero_rms=zero,
                      suffix_residual_rms=residual, suffix_wrong_sign_rms=wrong_sign,
                      suffix_offsets_b_minus_a=actual.tolist(), suffix_errors=(actual-estimate).tolist(),
                      suffix_wrong_sign_errors=(actual+estimate).tolist())
    result.update(valid=True, reason='features_only_not_identity', G=g, X=x, Xminus=wrong,
                  full_support=bool(full), common_scale=scale, cutoff_frame=int(cutoff),
                  prefix_indices=prefix, suffix_indices=suffix,
                  orientations=dict(ABA=aba, BAB=bab),
                  supported_rows=[dict(track=key[0], date=key[1], diagonal=diag_lookup[key]) for key in sorted(supported)],
                  boundary_brackets_discarded=len(brackets)-len(prefix)-len(suffix))
    return result


def synthetic_assertions():
    def rows(dates, delta=(0., 0.), widths=None, acceleration=0.):
        dates = np.asarray(dates)
        widths = np.full(len(dates), 2.) if widths is None else np.asarray(widths)
        center = np.column_stack((dates*.4+delta[0], dates*.2+delta[1]+acceleration*dates**2))
        wh = np.column_stack((widths, np.full(len(dates), 2.)))
        return np.column_stack((dates, np.ones(len(dates)), center-wh/2, wh))
    a, b = rows(np.arange(1, 32, 2)), rows(np.arange(2, 31, 2), (3., -1.))
    fixed_parts = pair_features(a, b)
    assert fixed_parts['full_support'] and fixed_parts['suffix_residual_rms'] < 1e-10
    assert fixed_parts['X'][1] > 0 and fixed_parts['Xminus'][1] < 0
    # Separate parallel bees generate identical raw observations and features.
    assert fixed_parts == pair_features(a.copy(), b.copy())
    reversed_roles = pair_features(b, a)
    np.testing.assert_allclose(reversed_roles['prefix_offset_b_minus_a'], [-3., 1.])
    np.testing.assert_allclose(fixed_parts['G'], reversed_roles['G'])
    np.testing.assert_allclose(fixed_parts['X'], reversed_roles['X'])
    singleton = pair_features(rows([1, 3]), rows([2], (3., -1.)))
    assert singleton['valid'] and not singleton['full_support']
    assert len(singleton['brackets']) == 1 and singleton['X'] == singleton['Xminus'] == [0., 0.]
    sparse = pair_features(rows([1, 20]), rows([10, 30]))
    assert not sparse['valid']
    assert not pair_features(rows([1, 2]), rows([2, 3]))['valid']
    curve = pair_features(rows(np.arange(1, 32, 2), acceleration=.2), rows(np.arange(2, 31, 2), acceleration=.2))
    assert curve['suffix_residual_rms'] > .1  # NOT automatic different-bee evidence.
    # Unique-supported-row median, not per-bracket duplicated weighting.
    scaled = pair_features(rows([1, 3, 5], widths=[2, 20, 2]), rows([2, 4], widths=[2, 2]))
    assert np.isclose(scaled['common_scale'], np.sqrt(8.))
    return dict(passed=True, schema=SCHEMA, brackets=len(fixed_parts['brackets']),
                full_support=fixed_parts['full_support'], X=fixed_parts['X'], Xminus=fixed_parts['Xminus'],
                singleton_valid=singleton['valid'], acceleration_residual=curve['suffix_residual_rms'],
                claim='synthetic math only; parallel separate bees remain observationally equivalent')


if __name__ == '__main__':
    print(synthetic_assertions())
