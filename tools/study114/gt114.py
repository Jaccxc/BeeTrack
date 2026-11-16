"""Measure same-bee returns with competing other-bee crossings using annotated turns."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from math import hypot, sqrt


def turns(gt_rows, polygon):
    """Read prepared GT. Dev IDs were already repaired by beetrack.prep."""
    from beetrack.evaluate import by_track, zone_tracks

    rows = by_track(gt_rows)
    result = []
    for track_id, (_, _, _, crossings) in zone_tracks(rows, polygon).items():
        for index, direction in crossings:
            before, after = rows[index - 1], rows[index]
            if before[1] != track_id or after[1] != track_id or not 0 < after[0] - before[0] <= 25:
                raise ValueError("Invalid GT change of side")
            result.append([int(track_id), int(before[0]), int(after[0]), 1 if direction == "in" else -1,
                           before[2:6].tolist(), after[2:6].tolist()])
    return sorted(result, key=lambda row: (row[2], row[0]))


def distance(a, b):
    center_dx = a[0] + a[2] / 2 - b[0] - b[2] / 2
    center_dy = a[1] + a[3] / 2 - b[1] - b[3] / 2
    scale = (sqrt(a[2] * a[3]) + sqrt(b[2] * b[3])) / 2
    return hypot(center_dx, center_dy) / scale


def opportunity(records):
    """Reference opportunities from GT positions, counted once for each actual turn."""
    ordered = sorted(records, key=lambda row: (row[2], row[0]))
    arrivals = [r[2] for r in ordered]
    per_bee = defaultdict(list)
    for i, (bee, u, t, sign, pre, post) in enumerate(ordered):
        if t <= u or sign not in (1, -1) or min(pre[2:] + post[2:]) <= 0:
            raise ValueError("Invalid ground-truth crossing")
        per_bee[bee].append(i)
    following = {i: j for ix in per_bee.values() for i, j in zip(ix, ix[1:])}
    stat = Counter()
    origins, mixed_bees = [], set()
    for i, (bee, _, t, sign, _, post) in enumerate(ordered):
        nxt = following.get(i)
        end = bisect_right(arrivals, t + 50)
        alternatives = [j for j in range(bisect_right(arrivals, t), end)
                        if ordered[j][3] == -sign and ordered[j][1] >= t
                        and distance(post, ordered[j][4]) <= 4]
        other = sum(ordered[j][0] != bee for j in alternatives)
        in_time = nxt is not None and 0 < ordered[nxt][2] - t <= 50 and ordered[nxt][1] >= t
        if nxt is not None and ordered[nxt][3] != -sign:
            raise ValueError("A GT segment's consecutive crossings must alternate")
        admitted = nxt in alternatives
        mixed = bool(admitted and other)
        stat.update(dict(any_next=nxt is not None, within_time=in_time, admitted_next=admitted,
                         mixed=mixed, other_present=bool(other)))
        if mixed:
            mixed_bees.add(bee)
        origins.append([bee, t, None if nxt is None else ordered[nxt][2], bool(in_time),
                        admitted, other])
    return dict(n_turns=len(ordered), n_bees=len(per_bee),
                gt_events=sum(len(ix) % 2 for ix in per_bee.values()),
                n_each_direction=[sum(r[3] == d for r in ordered) for d in (1, -1)],
                any_next=stat["any_next"], within_time=stat["within_time"],
                admitted_next=stat["admitted_next"], mixed=stat["mixed"],
                mixed_bees=len(mixed_bees), other_present=stat["other_present"], origins=origins)


def self_check():
    import numpy as np

    # Bee 11 enters and leaves. Bee 12 leaves between those observations.
    polygon = [[0,0],[20,0],[20,20],[0,20]]
    rows = np.array([[1,11,-3,6,4,4],[2,11,0,6,4,4],[5,11,-3,6,4,4],
                     [2,12,0,6,4,4],[3,12,-3,6,4,4]], float)
    crossed = turns(rows, polygon)
    assert [(r[0],r[2],r[3]) for r in crossed] == [(11,2,1),(12,3,-1),(11,5,-1)]
    check = opportunity(crossed)
    assert check["gt_events"] == 1 and check["mixed"] == check["mixed_bees"] == 1
    assert opportunity([])["n_turns"] == 0
    def row(bee, u, t, direction, x):
        return [bee,u,t,direction,[x,0,4,4],[x,0,4,4]]
    exact = [row(1,1,2,1,0),row(1,51,52,-1,16),row(2,2,3,-1,0)]
    assert opportunity(exact)["mixed"] == 1
    assert opportunity([exact[0],row(1,52,53,-1,16),exact[2]])["within_time"] == 0
    assert opportunity([exact[0],row(1,51,52,-1,16.01),exact[2]])["admitted_next"] == 0
    assert opportunity([exact[0],exact[1],row(2,1,3,-1,0)])["mixed"] == 0
    print("GT turn parity, time and distance boundaries, overlapping observations, and mixed identities verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["self-check"])
    parser.parse_args()
    self_check()
