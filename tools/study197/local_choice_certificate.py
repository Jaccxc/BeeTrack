"""Exact integer certificates for complete local trajectory alternatives.

r = all-low signed pooled window counts -3*GT; column i of V is the signed
window-count change of one WHOLE component alternative. A feasible x selects
complete columns (binary0/1), and its numerator is ||r+Vx||_1. The caller must
ensure alternatives/configurations are physically legal in the same movie;
this numerical tool cannot establish legality from count vectors alone.

The bound relaxes all physical coupling and omits F1: it is optimistic.
Re-evaluate F1 on the actual full-movie configuration with the official scorer;
never assemble TP values precomputed separately on alternative arms.
"""
import argparse
import itertools
import json
from numbers import Integral

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must contain integers, not floating-point values")
    return int(value)


def _inputs(r, V):
    r = np.asarray(r, dtype=object)
    V = np.asarray(V, dtype=object)
    if r.ndim != 1:
        raise ValueError("r must be a one-dimensional integer vector")
    if V.ndim == 1 and not V.size:
        V = V.reshape(len(r), 0)
    if V.ndim != 2 or V.shape[0] != len(r):
        raise ValueError("V must have one row per window and one column per whole component")
    residual = [_integer(v, "r") for v in r]
    columns = [[_integer(V[j, i], "V") for j in range(len(r))] for i in range(V.shape[1])]
    return residual, columns


def feasible_numerator(r, V, x):
    """Exact witness numerator; x must choose complete component columns."""
    residual, columns = _inputs(r, V)
    x = np.asarray(x, dtype=object)
    if x.shape != (len(columns),):
        raise ValueError("x must have one binary entry per column")
    x = [_integer(value, "x") for value in x]
    if any(value not in (0, 1) for value in x):
        raise ValueError("a witness must be binary, not a fractional LP configuration")
    changed = [residual[j] + sum(column[j]*bit for column, bit in zip(columns, x))
               for j in range(len(residual))]
    return sum(abs(value) for value in changed)


def certificate(r, V, p, q):
    """Exact bound ceil((p'r +sum_i min(0,p'v_i))/q), with |p_j|<=q.

    Python integers avoid overflow. No floating point is used in this proof,
    including the ceiling when the scaled numerator is negative.
    ||r+Vx||_1 >= p'(r+Vx)/q >= (p'r+sum min(0,p'v_i))/q;
    the feasible numerator is integral, so taking the ceiling is valid.
    """
    residual, columns = _inputs(r, V)
    q = _integer(q, "q")
    p = np.asarray(p, dtype=object)
    if q <= 0 or p.shape != (len(residual),):
        raise ValueError("q must be positive and p must match the window vector")
    p = [_integer(value, "p") for value in p]
    if any(abs(value) > q for value in p):
        raise ValueError("every multiplier must satisfy |p_j|<=q")
    linear = sum(multiplier*value for multiplier, value in zip(p, residual))
    dot_columns = [sum(multiplier*value for multiplier, value in zip(p, column)) for column in columns]
    negative = sum(min(0, value) for value in dot_columns)
    numerator = linear+negative
    lower = -((-numerator)//q)
    return dict(p=p, q=q, linear_term=linear, column_dot_products=dot_columns,
                negative_part_sum=negative, scaled_numerator=numerator, lower_bound=lower)


def lower_bound(r, V, p, q):
    return certificate(r, V, p, q)["lower_bound"]


def verify_certificate(r, V, proof):
    """Recompute every integer proof field, rejecting edited certificates."""
    expected = certificate(r, V, proof["p"], proof["q"])
    for field, value in expected.items():
        if field not in proof or proof[field] != value:
            raise ValueError(f"certificate field failed exact verification: {field}")
    return True


def lp_certificate(r, V, q=1_000_000):
    """Optional SciPy proposal only; quantize/clip and recheck exact p/q proof.

    LP's floating objective is diagnostic, never used as a proved lower bound.
    Restrict proposal denominator to an exactly represented float integer;
    manually supplied integer certificates have no such magnitude limit.
    """
    residual, columns = _inputs(r, V)
    q = _integer(q, "q")
    if not 1 <= q <= 2**52:
        raise ValueError("LP proposal q must lie in1..2**52")
    m, n = len(residual), len(columns)
    if not m or not n:
        p = [q*((value > 0)-(value < 0)) for value in residual]
        proof = certificate(r, V, p, q)
        verify_certificate(r, V, proof)
        return dict(**proof, proposal="empty exact case; no LP required", lp_relaxation=sum(abs(v) for v in residual))
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, eye, hstack, vstack
    try:
        rr = np.array(residual, float)
        matrix = np.array(columns, float).T
    except (OverflowError, ValueError) as error:
        raise ValueError("LP proposal cannot represent these integers; supply p/q manually") from error
    if not np.isfinite(rr).all() or not np.isfinite(matrix).all():
        raise ValueError("LP proposal cannot represent these integers; supply p/q manually")
    matrix = csr_matrix(matrix)
    identity = eye(m, format="csr")
    A = vstack([hstack([matrix, -identity]), hstack([-matrix, -identity])], format="csr")
    solution = linprog(np.r_[np.zeros(n), np.ones(m)], A_ub=A, b_ub=np.r_[-rr, rr],
                       bounds=[(0, 1)]*n+[(0, None)]*m, method="highs-ds")
    if not solution.success:
        raise RuntimeError("optional LP proposal failed: "+solution.message)
    dual = solution.ineqlin.marginals
    y = dual[m:]-dual[:m]
    if not np.isfinite(y).all():
        raise RuntimeError("optional LP produced nonfinite multipliers")
    p = [max(-q, min(q, int(round(float(value)*q)))) for value in y]
    proof = certificate(r, V, p, q)
    verify_certificate(r, V, proof)
    return dict(**proof, proposal="SciPy dual -> rounded/clipped integers -> exact revalidation",
                lp_relaxation=float(solution.fun))


def self_test():
    # Two opposite whole-event date relocations between two disjoint windows.
    # GT pooled3*[1,1], all-low counts[4,2]; every configuration is nonnegative.
    r, V = [1, -1], [[-1, 1], [1, -1]]
    physical = {"00": feasible_numerator(r, V, [0, 0]), "10": feasible_numerator(r, V, [1, 0]),
                "01": feasible_numerator(r, V, [0, 1]), "11": feasible_numerator(r, V, [1, 1])}
    assert physical == {"00": 2, "10": 0, "01": 4, "11": 2}
    for choice in itertools.product((0, 1), repeat=2):
        predicted = np.array([4, 2]) + np.array(V) @ choice
        assert (predicted >= 0).all()
        assert int(np.abs(predicted-3*np.array([1, 1])).sum()) == feasible_numerator(r, V, choice)
    assert lower_bound(r, V, [0, 0], 1) == 0
    # Binary optimum1 but x=.5 has LP numerator0: relaxation is not a witness.
    gap = lp_certificate([1], [[-2]])
    assert gap["lower_bound"] == 0 and gap["lp_relaxation"] == 0
    assert min(feasible_numerator([1], [[-2]], [x]) for x in (0, 1)) == 1
    assert lower_bound([-1], [], [1], 3) == 0
    assert lower_bound([-4], [], [1], 3) == -1
    assert feasible_numerator([], np.empty((0, 3), int), [1, 0, 1]) == 0
    assert lower_bound([], np.empty((0, 3), int), [], 7) == 0
    assert feasible_numerator([2, -3], [], []) == 5
    assert lower_bound([2, -3], [], [5, -5], 5) == 5
    assert lower_bound([2, -3], [[0, 0], [0, 0]], [5, -5], 5) == 5
    huge = 10**60+1
    assert feasible_numerator([huge], [], []) == lower_bound([huge], [], [3], 3) == huge
    for invalid in (([1], [[-2]], [.5]), ([1], [[-2]], [2])):
        try:
            feasible_numerator(*invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("nonbinary witness accepted")
    try:
        lower_bound([1], [[-2]], [3], 2)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-box multiplier accepted")
    edited = dict(gap, lower_bound=1)
    try:
        verify_certificate([1], [[-2]], edited)
    except ValueError:
        pass
    else:
        raise AssertionError("edited proof accepted")
    rng = np.random.default_rng(19503)
    checked, lp_checked = 0, 0
    for m in range(5):
        for n in range(7):
            for repeat in range(8):
                r = rng.integers(-5, 6, m)
                V = rng.integers(-3, 4, (m, n))
                choices = list(itertools.product((0, 1), repeat=n))
                optimum = min(feasible_numerator(r, V, x) for x in choices)
                witness = feasible_numerator(r, V, choices[int(rng.integers(len(choices)))])
                q = int(rng.integers(1, 18))
                proof = certificate(r, V, rng.integers(-q, q+1, m), q)
                assert verify_certificate(r, V, proof)
                assert proof["lower_bound"] <= optimum <= witness
                checked += 1
                if repeat == 0:
                    proof = lp_certificate(r, V, q=10000)
                    assert proof["lower_bound"] <= optimum <= witness
                    assert verify_certificate(r, V, proof)
                    lp_checked += 1
    return dict(random_seed=19503, exhaustive_random_cases=checked, quantized_LP_cases=lp_checked,
                opposite_whole_options=physical, LP_gap=dict(lower=0, binary_optimum=1),
                negative_ceiling="ceil(-1/3)=0; ceil(-4/3)=-1", empty_zero_columns="passed",
                integer_overflow="Python integer10**60+1 exact", tampered_proof="rejected",
                limitation="Optimistic bound omits F1/physical coupling. Official full-movie F1 must be rescored for the actual configuration; never splice alternative-arm TP.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        parser.print_help()
