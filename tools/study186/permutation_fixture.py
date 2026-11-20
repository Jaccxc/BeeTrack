"""Private CPU-only synthetic checks; run from final_project with uv run python <this file>.

No dataset sequence is loaded. All rows are complete 10-column MOT rows.
Binary side 1 means polygon-inside. Assertions use the project's actual scorer.
"""
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path.cwd()
assert (ROOT/'beetrack/p0.py').is_file(), 'Run from the BeeTrack repository root.'
sys.path.insert(0, str(ROOT))
from beetrack.p0 import events, totals, window_err

POLY = [[0, 0], [100, 0], [100, 100], [0, 100]]


def endpoint_counts(starts, ends):
    return (sum((1-s)*e for s, e in zip(starts, ends)),
            sum(s*(1-e) for s, e in zip(starts, ends)))


def row(frame, identity, side, lane=0):
    # Centres are strictly inside or outside, never on the polygon boundary.
    return [frame, identity, 45 if side else 115, 15+20*lane, 10, 10, 1, -1, -1, -1]


def exhaustive():
    permutations_checked = swaps_checked = 0
    for n in range(1, 6):
        for starts in itertools.product((0, 1), repeat=n):
            for ends in itertools.product((0, 1), repeat=n):
                for perm in itertools.permutations(ends):
                    incoming, outgoing = endpoint_counts(starts, perm)
                    assert incoming-outgoing == sum(ends)-sum(starts)
                    permutations_checked += 1
    for s1, s2, e1, e2 in itertools.product((0, 1), repeat=4):
        direct = endpoint_counts((s1, s2), (e1, e2))
        swapped = endpoint_counts((s1, s2), (e2, e1))
        delta = (s2-s1)*(e1-e2)
        assert tuple(a-b for a, b in zip(direct, swapped)) == (delta, delta)
        # Also verify the endpoint formula against official events on full rows.
        for ends in ((e1, e2), (e2, e1)):
            rows = [row(f, i+1, s if f == 1 else e, i)
                    for i, (s, e) in enumerate(zip((s1, s2), ends)) for f in (1, 2)]
            assert tuple(totals(events(rows, POLY))) == endpoint_counts((s1, s2), ends)
        swaps_checked += 1
    return dict(permutations=permutations_checked, binary_swaps=swaps_checked)


def timing_rows(swap):
    rows = []
    for f in range(1, 1601):
        sides = [int(f >= 100 and not 900 <= f < 1000), int(f >= 700)]
        for lane, side in enumerate(sides):
            identity = 2-lane if swap and f > 800 else lane+1
            rows.append(row(f, identity, side, lane))
        rows.append(row(f, 3, int(f < 1100), 2))
    return rows


def neutral_rows(swap):
    rows = []
    for f in range(1, 1001):
        if f == 501:
            rows.append(row(f, 9, 0))  # one merged observation, neutral isolated ID
            continue
        for lane, side in enumerate((0, int(400 <= f <= 700))):
            identity = 2-lane if swap and f > 501 else lane+1
            rows.append(row(f, identity, side, lane))
    return rows


def main():
    checks = exhaustive()
    a, b = (events(timing_rows(swap), POLY) for swap in (False, True))
    assert a.tolist() == [[1000., 1.], [700., 1.], [1100., -1.]]
    assert b.tolist() == [[100., 1.], [1000., 1.], [1100., -1.]]
    assert (totals(a) == totals(b)).all() and totals(a).tolist() == [2, 1]
    assert window_err(b, a, 1600, win=None) == 0
    assert np.isclose(window_err(b, a, 1600), 1.6)
    local, whole = [], []
    for swap in (False, True):
        rows = neutral_rows(swap)
        local.append(events([r for r in rows if 499 <= r[0] <= 503], POLY))
        whole.append(events(rows, POLY))
    assert local[0].tolist() == []
    assert local[1].tolist() == [[502., 1.], [502., -1.]]
    assert all(e.tolist() == [] for e in whole)
    print(json.dumps(dict(checks=checks,
        timing=dict(direct=a.tolist(), swapped=b.tolist(), totals=totals(a).tolist(),
                    direct_W_percent=0, swapped_W_percent=100*window_err(b, a, 1600)/len(a),
                    swapped_raw_error=window_err(b, a, 1600), whole_video_error=window_err(b, a, 1600, win=None)),
        merge_split=dict(local_direct=local[0].tolist(), local_swapped=local[1].tolist(),
                         whole_direct=whole[0].tolist(), whole_swapped=whole[1].tolist())), indent=2))


if __name__ == "__main__":
    main()
