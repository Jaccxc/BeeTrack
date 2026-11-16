"""Best attainable window count error over pairwise track-disjoint suffix swaps."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import bmat, coo_matrix, csc_matrix, eye


def earliest_disjoint(candidates):
    """Return candidate indices in cut-frame order, breaking ties by sorted track IDs."""
    used = set()
    selected = []
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i][2], *sorted(candidates[i][:2])))
    for i in order:
        a, b, _ = candidates[i]
        if a == b:
            raise ValueError("a swap needs two distinct tracks")
        if a not in used and b not in used:
            selected.append(i)
            used.update((a, b))
    return selected


def best_disjoint(base_counts, gt_counts, candidates, deltas, weights):
    """Return (selected candidate indices, minimum weighted L1 count error).

    Each per-window weight multiplies both in and out errors. Swaps on disjoint
    tracks have additive event-count deltas when scored on the same windows.
    """
    base = np.asarray(base_counts, dtype=float)
    truth = np.asarray(gt_counts, dtype=float)
    changes = np.asarray(deltas, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if base.ndim != 2 or base.shape[1] != 2 or truth.shape != base.shape:
        raise ValueError("base and ground truth must have matching (windows, 2) shapes")
    if weights.shape not in {(len(base),), (len(base), 1)} or not np.all(weights > 0):
        raise ValueError("provide one positive weight per window")
    if any(not np.isfinite(arr).all() for arr in (base, truth, changes, weights)):
        raise ValueError("counts, deltas and weights must be finite")
    if not len(candidates) and changes.shape == (0,):
        changes = changes.reshape((0, *base.shape))
    if changes.shape != (len(candidates), *base.shape):
        raise ValueError("provide one (windows, 2) count delta per candidate")
    if any(a == b for a, b, _ in candidates):
        raise ValueError("a swap needs two distinct tracks")

    residual = (base - truth).ravel()
    cell_weights = np.repeat(weights.ravel(), 2)
    # Removing zero-delta switches preserves the attainable optimum: any such
    # switch can be omitted from a feasible subset without changing its score.
    active = np.flatnonzero(np.any(changes != 0, axis=(1, 2)))
    if not active.size or not np.any(residual):
        return [], float(cell_weights @ np.abs(residual))

    num_swaps = len(active)
    num_cells = len(residual)
    ids = [track for i in active for track in candidates[i][:2]]
    tracks, row_indices = np.unique(ids, return_inverse=True)
    shared_tracks = coo_matrix(
        (np.ones(2 * num_swaps), (row_indices, np.repeat(np.arange(num_swaps), 2))),
        shape=(len(tracks), num_swaps),
    )
    delta_matrix = csc_matrix(changes[active].reshape(num_swaps, num_cells).T)
    # Variables x choose swaps. Auxiliary cell errors u bound both signs of
    # residual + D x. Each row of shared_tracks enforces at most one use of an ID.
    minus_identity = -eye(num_cells, format="csc")
    matrix = bmat(
        [[shared_tracks, None], [delta_matrix, minus_identity], [-delta_matrix, minus_identity]],
        format="csc",
    )
    upper = np.r_[np.ones(len(tracks)), -residual, residual]
    result = milp(
        np.r_[np.zeros(num_swaps), cell_weights],
        integrality=np.r_[np.ones(num_swaps), np.zeros(num_cells)],
        bounds=Bounds(0, np.r_[np.ones(num_swaps), np.full(num_cells, np.inf)]),
        constraints=LinearConstraint(matrix, -np.inf, upper),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.status != 0:
        raise RuntimeError(f"no certified optimum: {result.message}")

    binary = result.x[:num_swaps]
    selected = active[binary > 0.5].tolist()
    used = [track for i in selected for track in candidates[i][:2]]
    if not np.allclose(binary, np.rint(binary), rtol=0, atol=1e-7) or len(used) != len(set(used)):
        raise RuntimeError("solver did not return an integral, track-disjoint selection")
    score = float(cell_weights @ np.abs(residual + changes[selected].sum(axis=0).ravel()))
    if not np.allclose([result.fun, result.mip_dual_bound], score, atol=1e-7, rtol=1e-9):
        raise RuntimeError(f"uncertified score {score}, objective {result.fun}, bound {result.mip_dual_bound}")
    return selected, score


if __name__ == "__main__":
    counts = np.array([[1, 1], [1, 0]])
    truth = np.zeros_like(counts)
    swaps = [(10, 20, 5), (10, 30, 7), (40, 50, 11)]
    deltas = np.array([[[-1, 0], [0, 0]], [[0, 0], [-1, 0]], [[0, -1], [0, 0]]])
    weights = [1, 2]
    assert earliest_disjoint(swaps) == [0, 2]
    assert best_disjoint(counts, truth, [], [], weights) == ([], 4)
    selected, optimum = best_disjoint(counts, truth, swaps, deltas, weights)
    assert selected == [1, 2] and optimum == 1
    print("Disjoint swap synthetic checks passed.")
