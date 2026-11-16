"""Certified window-count oracle. Its optional donor events share one selection across all offsets."""
import argparse
from itertools import combinations
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog
from scipy.sparse import csr_matrix, eye, hstack


def _signed_frames(values, nframes):
    frames = np.asarray(values)
    if frames.ndim != 1 or (frames.size and frames.dtype.kind not in "iu"):
        raise ValueError("Expected a 1-D sequence of signed integer, 1-based frame numbers.")
    if np.any(frames > nframes) or np.any(frames < -nframes) or np.any(frames == 0):
        raise ValueError("Event outside the clip.")
    return frames.astype(np.int64)


def _incidence(frames, nframes):
    """One sparse binary column per event; each intersects one cell at each of the 30 offsets."""
    offsets = np.arange(0, 750, 25, dtype=np.int64)
    sizes = 2 * ((nframes + offsets + 749) // 750)
    starts = np.cumsum(sizes) - sizes
    rows = starts[:,None] + 2*((np.abs(frames)[None,:]-1 + offsets[:,None])//750) + (frames < 0)
    columns = np.tile(np.arange(len(frames)), len(offsets))
    return csr_matrix((np.ones(rows.size, dtype=np.int64), (rows.ravel(),columns)),
                      shape=(int(sizes.sum()),len(frames)))


def _counts(matrix):
    return np.asarray(matrix.sum(axis=1)).ravel()


def _interval_floor(target, available):
    return int((np.maximum(-target, 0) + np.maximum(target - available, 0)).sum())


def _round_dual(dual, residual, incidence, scale):
    # |y|<=1 supplies an independent lower bound:
    # y*(GT-base) - sum_j max(0,(A'y)_j) <= min_{0<=z<=1} ||GT-base-Az||_1.
    # Use exact integer arithmetic after quantizing y. Ceil is valid for binary z.
    y = np.rint(np.clip(dual,-1,1) * scale).astype(np.int64)
    weights = incidence.T @ y
    numerator = sum(int(b)*int(v) for b,v in zip(residual,y)) - sum(max(0,int(w)) for w in weights)
    return -((-numerator)//scale), numerator, y


def score_oracle(base_frames, donor_frames, gt_frames, nframes, *, time_limit=120.0):
    """Return exact L1 numerators (percentage denominator = 30 * number of GT events).

    Indices refer to the caller's original donor order, including equal-time events.
    GT times are used for this diagnostic; the witness is not a deployable selector.
    """
    if not isinstance(nframes,(int,np.integer)) or nframes < 1 or time_limit <= 0:
        raise ValueError("Clip length and solver time limit must be positive.")
    base, donor, gt = [_signed_frames(s,nframes) for s in (base_frames,donor_frames,gt_frames)]
    # By direction then time, each signed window is a consecutive interval of columns.
    # This matrix is totally unimodular, as is [A,I,-I]. Integer RHS and box bounds
    # admit an integral optimal vertex of the LP. The dual witness below checks the result.
    order = np.lexsort((np.arange(len(donor)),np.abs(donor),donor<0))
    a = _incidence(donor[order],nframes)
    base_cells, gt_cells = [_counts(_incidence(s,nframes)) for s in (base,gt)]
    residual = gt_cells-base_cells
    donor_cells = _counts(a)
    totals = {name: [int((s>0).sum()),int((s<0).sum())]
              for name,s in (("base",base),("donor",donor),("gt",gt))}
    dir_floor = 30*_interval_floor(np.subtract(totals["gt"],totals["base"]), np.array(totals["donor"]))
    cell_floor = _interval_floor(residual,donor_cells)
    native = int(abs(residual).sum())
    all_donors = int(abs(residual-donor_cells).sum())
    lower, upper = cell_floor, min(native,all_donors)
    if not dir_floor <= lower <= upper:
        raise AssertionError("Interval lower bounds contradict the endpoint selections.")
    chosen = np.zeros(len(donor),dtype=np.int8) if native <= all_donors else np.ones(len(donor),dtype=np.int8)
    proof = {"kind":"interval_floor"}
    solver_status = "interval floor attained"
    if lower < upper:
        m,p = a.shape
        ident = eye(m,format="csr")
        system = hstack([a,ident,-ident],format="csr")
        cost = np.r_[np.zeros(p),np.ones(2*m)]
        bounds = [(0,1)]*p + [(0,None)]*(2*m)
        with warnings.catch_warnings():
            # SciPy passes this HiGHS option through, with a version-dependent warning.
            warnings.filterwarnings("ignore",category=OptimizeWarning,message="Unrecognized options.*threads.*")
            result = linprog(cost,A_eq=system,b_eq=residual,bounds=bounds,method="highs-ds",
                             options={"threads":1,"time_limit":float(time_limit)})
        solver_status = str(result.message)
        if result.x is not None and np.all(np.isfinite(result.x[:p])):
            candidate = np.rint(np.clip(result.x[:p],0,1)).astype(np.int8)
            candidate_obj = int(abs(residual-a@candidate).sum())
            if candidate_obj < upper:
                upper, chosen = candidate_obj, candidate
        dual = result.eqlin.marginals
        if dual is not None and np.all(np.isfinite(dual)):
            for scale in (1, 1048576):
                cert, raw, y = _round_dual(dual,residual,a,scale)
                if cert > lower:
                    lower = cert
                    proof = dict(kind="rational_dual", scale=scale, numerator=raw,
                                 nonzero_weights=[[int(i),int(y[i])] for i in np.flatnonzero(y)])
                if upper == lower:
                    break
    if upper < lower or int(abs(residual-a@chosen).sum()) != upper:
        raise AssertionError("Inconsistent primal witness or lower-bound certificate.")
    picks = sorted(map(int,order[chosen.astype(bool)]))
    if int(abs(residual-_counts(_incidence(donor[picks],nframes))).sum()) != upper:
        raise AssertionError("Original-order donor witness did not replay.")
    certifies = upper == lower
    den = 30*len(gt)
    nums = dict(base_numerator=native,all_numerator=all_donors,direction_floor_numerator=dir_floor,
                cell_floor_numerator=cell_floor,optimal_numerator=upper if certifies else None)
    percent = {key.removesuffix("_numerator"):100*value/den if den and value is not None else None
               for key,value in nums.items()}
    return dict(nums,denominator=den,percent=percent,direction_totals=totals,
                selected_donor_indices=picks,primal_upper=upper,integer_lower=lower,integer_gap=upper-lower,
                certified=certifies,proof=proof,solver_status=solver_status)


def _reference(frames, nframes):
    # Independent direct event bucketing for synthetic exhaustive enumeration.
    bins = []
    for o in range(30):
        part = [[0,0] for _ in range((nframes+25*o+749)//750)]
        for f in frames:
            part[(abs(f)-1+25*o)//750][int(f<0)] += 1
        bins.extend(part)
    return np.array(bins)


def self_check():
    examples = [(750,[],[401],[1]),(1501,[-1,750],[-750,-750,751,1501],[751,-749,-1501]),
                (1,[1],[],[-1,1]),(3950,[],[],[]),(651,[],[-1,2],[]),
                (1750,[1,1,-25],[25,25,-31,-1,1750],[25,700,750,751,-750])]
    rng = np.random.default_rng(110)
    for n in (24,25,749,750,751,1000,3950):
        for _ in range(5):
            def sample(k):
                return list(map(int,rng.integers(1,n+1,k)*rng.choice([-1,1],k)))
            examples.append((n,sample(5),sample(6),sample(8)))
    dp=0
    for n,base,donor,truth in examples:
        solution=score_oracle(base,donor,truth,n)
        gt=_reference(truth,n)
        best=min(int(abs(_reference(base+[donor[i] for i in subset],n)-gt).sum())
                 for k in range(len(donor)+1) for subset in combinations(range(len(donor)),k))
        assert solution["certified"] and solution["optimal_numerator"] == best
        assert solution["denominator"]==30*len(truth)
        assert solution["base_numerator"] == int(abs(_reference(base,n)-gt).sum())
        assert solution["all_numerator"] == int(abs(_reference(base+donor,n)-gt).sum())
        assert solution["direction_floor_numerator"]<=solution["cell_floor_numerator"]<=best
        if solution["proof"]["kind"]=="rational_dual":
            dp+=1
        if n==750 and donor==[401]:
            assert (solution["direction_floor_numerator"],solution["cell_floor_numerator"],best)==(0,16,30)
    assert dp>0
    print(f"{len(examples)} exhaustive synthetic cases passed; {dp} used an integer-certified LP dual.")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command",choices=["self-check"])
    cli.parse_args()
    self_check()
