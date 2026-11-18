"""Box counting rule of Lei et al. (2024, Insects 15(12):974, "A Honey Bee In-and-Out Counting Method Based on Multiple
Object Tracking Algorithm"), the counting baseline next to our gate rule (evaluate.zone_tracks).
Their rule, per frame and per tracker ID: a crossing of the detection box into it adds 1 to In and puts the ID in
Already-counted; a crossing out of it adds 1 to Out, except when the ID is in Already-counted (the bee only passed
through the box): then the ID is removed and In goes down by 1. The paper leaves open whether that exit also adds 1 to
Out; the default here does not (that reading is the one that fixes the pass-through error the rule was made for),
literal=True also adds it. Here the box is our entrance polygon (zones.json), a crossing is a change of side of the box
centre between two boxes of one ID in a row, and IDs are the tracker's own (no split at long gaps).
Differences from the gate rule, per ID: a track that starts inside, leaves and comes back counts out + in here, 0 there;
an out is dated at the first exit here, the last one there."""
import numpy as np

from .evaluate import zone_tracks


def box_events(rows, poly, literal=False):
    """[(id, +1 in / -1 out, row of rows at the crossing)] for rows (frame, id, x, y, w, h) sorted by id, then frame."""
    out = []
    for tid, (_, _, _, cross) in zone_tracks(rows, poly).items():
        ev = []
        for j, d in cross:
            if d == "in":
                ev.append((tid, 1, j))
            elif ev and ev[-1][1] == 1:    # in Already-counted: remove the ID, In - 1 (the entry is undone)
                ev.pop()
                if literal:
                    ev.append((tid, -1, j))
            else:
                ev.append((tid, -1, j))
        out += ev
    return out


def _check():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    paths = {1: [-5, 5, 20], 2: [5, 20], 3: [-5, 5], 4: [5, 20, 5], 5: [-5, 5, 20, 5], 6: [20, 30]}   # x of the centre, frame by frame
    rows = np.array([[f, i, x, 5, 0, 0] for i, xs in paths.items() for f, x in enumerate(xs, 1)], float)
    got = lambda lit: [(i, d) for i, d, _ in box_events(rows, sq, lit)]
    assert got(False) == [(2, -1), (3, 1), (4, -1), (4, 1), (5, 1)], got(False)   # pass-through 1 and outside 6 count 0
    assert got(True) == [(1, -1), (2, -1), (3, 1), (4, -1), (4, 1), (5, -1), (5, 1)], got(True)
    assert [j for i, d, j in box_events(rows, sq) if i == 5] == [13]               # the last entry survives


_check()
