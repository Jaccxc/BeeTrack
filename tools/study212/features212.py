"""Pure212 original-net-event phase features, no image/GT/artifact IO.

One ORIGINAL canonical gap run, dates+inside bits only. Original >25 gap is
invalid (caller must resolve canonical runs first); each decimated phase is
re-split at actual observed >25 gaps, no min3 filter, no interpolation.
All O/D/T denominators are k INCLUDING empty phases. Random controls permute
phase LABELS among fixed original dates, retaining exact counts per phase.
"""
import hashlib
import numpy as np


SCHEMA = 'phase212.features.v1'
FEATURE_SCHEMA = SCHEMA
G_NAMES = ('log1p_n', 'n_over_inclusive_span', 'log1p_side_flips',
           'min_endpoint_same_side_run_over_n', 'min_inside_outside_over_n',
           'max_original_gap_over_25', 'direction')
PHASE_NAMES = ('k2_O', 'k2_D', 'k2_T', 'k3_O', 'k3_D', 'k3_T')


def _input(frames, inside):
    raw = np.asarray(frames)
    bits = np.asarray(inside)
    if raw.ndim != 1 or bits.shape != raw.shape or bits.dtype != np.bool_:
        raise ValueError('frames[N] and explicit boolean inside[N] required; unknown bits must not be coerced')
    if (not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)
            or not np.isfinite(raw).all() or (raw < 1).any()
            or (raw != np.floor(raw)).any() or (raw >= 2**53).any()):
        raise ValueError('positive exactly representable integral native timestamps required')
    order = np.argsort(raw, kind='stable')
    dates = raw[order].astype(np.int64)
    if (np.diff(dates) <= 0).any():
        raise ValueError('duplicate native timestamp/alias within original run')
    return dates, bits[order]


def event_summary(dates, bits):
    """A real subsegment contributes at most one net event, dated on a real row."""
    if len(dates) < 2 or bits[0] == bits[-1]:
        return None
    flips = np.flatnonzero(bits[1:] != bits[:-1])+1
    direction = 1 if bits[-1] else -1
    same_direction = flips[bits[flips] == bits[-1]]
    return dict(direction=direction, date=int(dates[same_direction[-1]]))


def phase_summary(dates, bits, labels, k, original_direction, original_date):
    """Count phases with observed2+ segment, unique same-direction event, date±5."""
    totals = np.zeros(3, dtype=np.int64)
    evidence = []
    for phase in range(k):
        selected = np.flatnonzero(labels == phase)
        # Dates remain fixed/chronological, including after label permutation.
        cut = np.r_[0, np.flatnonzero(np.diff(dates[selected]) > 25)+1, len(selected)]
        groups = [selected[a:b] for a, b in zip(cut[:-1], cut[1:]) if b > a]
        observed = any(len(group) >= 2 for group in groups)
        events = [event for group in groups if (event := event_summary(dates[group], bits[group])) is not None]
        directional = len(events) == 1 and events[0]['direction'] == original_direction
        temporal = directional and abs(events[0]['date']-original_date) <= 5
        totals += (int(observed), int(directional), int(temporal))
        evidence.append(dict(phase=phase, dates=dates[selected].tolist(),
                             segments=[dates[group].tolist() for group in groups],
                             events=events, O=bool(observed), D=bool(directional), T=bool(temporal)))
    return totals.astype(float)/k, evidence


def control_seed(dates, k, salt):
    text = 'phase212-v1|'+str(k)+'|'+str(salt)+'|'+','.join(str(int(date)) for date in dates)
    return int.from_bytes(hashlib.sha256(text.encode('ascii')).digest()[:8], 'big')


def run_features(frames, inside):
    """JSON-safe G7/R6/P6/T6, with T=P-R (T vector differs from temporal item).

    No-net-event/empty input returns validFalse, no numeric candidate. Canonical
    run crossing an original >25 gap returns invalid, never silently joins aliases.
    Caller owns polygon inside conversion/canonicalrun labels, not this module.
    """
    dates, bits = _input(frames, inside)
    result = dict(schema=SCHEMA, valid=False, reason=None, G_names=list(G_NAMES),
                  phase_names=list(PHASE_NAMES), G=None, R=None, R_alt=None, P=None, T=None)
    if not len(dates):
        result['reason'] = 'empty_original_run'; return result
    gaps = np.diff(dates)
    if (gaps > 25).any():
        result['reason'] = 'original_run_has_unresolved_gap_alias'; return result
    original = event_summary(dates, bits)
    if original is None:
        result['reason'] = 'not_original_net_event'; return result
    n = len(dates)
    flips = np.flatnonzero(bits[1:] != bits[:-1])+1
    prefix_same = int(flips[0])
    suffix_same = n-int(flips[-1])
    inside_count = int(bits.sum())
    g = [float(np.log1p(n)), float(n/(int(dates[-1])-int(dates[0])+1)),
         float(np.log1p(len(flips))), float(min(prefix_same, suffix_same)/n),
         float(min(inside_count, n-inside_count)/n), float(gaps.max()/25), float(original['direction'])]
    p, r, r_alt, phase_evidence, random_evidence = [], [], [], {}, {}
    for k in (2, 3):
        labels = (dates-1) % k
        actual, evidence = phase_summary(dates, bits, labels, k, original['direction'], original['date'])
        randomized, random_records = [], []
        for salt in range(16):
            seed = control_seed(dates, k, salt)
            perm = np.random.Generator(np.random.PCG64(seed)).permutation(n)
            shuffled = labels[perm]
            assert np.array_equal(np.bincount(labels, minlength=k), np.bincount(shuffled, minlength=k))
            values, details = phase_summary(dates, bits, shuffled, k, original['direction'], original['date'])
            randomized.append(values)
            random_records.append(dict(salt=salt, seed=seed, permutation=perm.tolist(), labels=shuffled.tolist(),
                                       ODT=values.tolist(), phases=details))
        p.extend(actual.tolist())
        r.extend(np.mean(randomized[:8], axis=0).tolist())
        r_alt.extend(np.mean(randomized[8:], axis=0).tolist())
        phase_evidence[str(k)] = evidence
        random_evidence[str(k)] = random_records
    result.update(valid=True, reason='original_net_event_features_only', G=g, P=p, R=r, R_alt=r_alt,
                  T=(np.asarray(p)-np.asarray(r)).tolist(), original_event=original,
                  dates=dates.tolist(), inside=bits.tolist(), phases=phase_evidence, controls=random_evidence)
    return result


def synthetic_assertions():
    # Sustained same trajectory; both phases retain observable directional event.
    sustained = run_features(np.arange(1, 19), np.arange(1, 19) >= 9)
    assert sustained['valid'] and sustained['P'] == [1., 1., 1., 1., 1., 1.]
    # Alternating observations from two stationary sources straddling boundary:
    # original false net1->inside; fixed parity phases contain no real event.
    alternating = run_features(np.arange(1, 19), np.arange(1, 19) % 2 == 0)
    assert alternating['valid'] and alternating['P'][:3] == [1., 0., 0.]
    # Original gaps<=25; a decimated phase's >25 gap must split into singletons.
    split = run_features([1, 14, 27, 40], np.array([False, False, True, True]))
    assert split['valid']
    assert split['P'][:3] == [0., 0., 0.]
    # Two original observations are a legitimate net event without a min3 gate;
    # singleton phase groups are retained in evidence but do not invent events.
    tiny = run_features([1, 2], np.array([False, True]))
    assert tiny['valid'] and tiny['P'] == [0.]*6
    alias = run_features([1, 27], np.array([False, True]))
    assert not alias['valid']
    for dates, flags in (([1, 1], np.array([False, True])), ([1, 2], np.array([0, -1])),
                         ([1., 2.5], np.array([False, True]))):
        try:
            run_features(dates, flags)
        except ValueError:
            pass
        else:
            raise AssertionError('alias/unknown/nonintegral date accepted')
    # Exact observed date difference5 passes,6 fails; no interpolated timestamp.
    d = np.array([1, 6]); bits = np.array([False, True]); labels = np.array([0, 0])
    five, _ = phase_summary(d, bits, labels, 2, 1, 11)
    six, _ = phase_summary(d, bits, labels, 2, 1, 12)
    assert five.tolist() == [.5, .5, .5] and six.tolist() == [.5, .5, 0.]
    assert sustained == run_features(np.arange(18, 0, -1), (np.arange(1, 19) >= 9)[::-1])
    assert len(sustained['R_alt']) == 6
    for k in ('2', '3'):
        controls = sustained['controls'][k]
        assert [row['salt'] for row in controls] == list(range(16))
        index = 0 if k == '2' else 3
        np.testing.assert_array_equal(np.mean([row['ODT'] for row in controls[8:]], axis=0), sustained['R_alt'][index:index+3])
    return dict(passed=True, sustained_P=sustained['P'], alternating_P=alternating['P'],
                decimated_gap_P=split['P'], singleton_P=tiny['P'], seed_k2_salt0=control_seed(np.arange(1, 19), 2, 0))


if __name__ == '__main__':
    print(synthetic_assertions())
