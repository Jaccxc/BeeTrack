"""Verify §110 with exhaustive per-direction subsets and integer certificates.

Run from the repository root with:
    uv run python tools/study110/verify_witness.py docs/study110_replay.json.gz
Recomputes fixed-event counting windows from signed one-based times in the archive.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np



def intervals(n):
    for origin in range(0,750,25):
        for left in range(1-origin,n+1,750):
            yield max(1,left), min(n+1,left+750)


def cells(base, donor, truth, n):
    residual, design = [], []
    for lo,hi in intervals(n):
        for sign in (1,-1):
            count = lambda events: sum(lo <= sign*f < hi for f in events)
            residual.append(count(truth)-count(base))
            design.append([int(lo <= sign*f < hi) for f in donor])
    return residual, design


def exhaustive(residual, design, donor, max_dim=18):
    """Directions separate; enumerate each feasible set in batches, compress equal rows."""
    value = 0
    tested = []
    for ix, sign in enumerate((1,-1)):
        active = [j for j,f in enumerate(donor) if sign*f>0]
        if len(active)>max_dim:
            return None,None
        if not active:
            value += sum(abs(r) for r in residual[ix::2])
            tested.append(1)
            continue
        repeated = Counter((r, *(row[j] for j in active))
                           for r,row in zip(residual[ix::2],design[ix::2]))
        unique = np.asarray(list(repeated),dtype=np.int32)
        weights = np.asarray(list(repeated.values()),dtype=np.int64)
        best = None
        for start in range(0,2**len(active),2048):
            block = np.arange(start,min(start+2048,2**len(active)))[:,None]
            patterns = ((block >> np.arange(len(active))) & 1).astype(np.int32)
            errors = np.abs(unique[:,0] - patterns @ unique[:,1:].T) @ weights
            new = int(errors.min())
            best = new if best is None else min(best,new)
        value += best
        tested.append(2**len(active))
    return value,tested


def verify(base, donors, truth, n, score):
    residual, a = cells(base,donors,truth,n)
    assert all(sum(row[j] for row in a)==30 for j in range(len(donors)))
    selected = score["selected_donor_indices"]
    assert len(selected) == len(set(selected)) and all(0<=i<len(donors) for i in selected)
    numerator = lambda ids: sum(abs(r-sum(row[j] for j in ids)) for r,row in zip(residual,a))
    assert numerator([])==score["base_numerator"]
    assert numerator(range(len(donors)))==score["all_numerator"]
    witness = numerator(selected)
    assert witness==score["primal_upper"]
    cell_floor=sum(max(0,-r,r-sum(row)) for r,row in zip(residual,a))
    assert score["cell_floor_numerator"]==cell_floor
    direct=sum(max(sum(sign*f>0 for f in base)-sum(sign*f>0 for f in truth),
                   sum(sign*f>0 for f in truth)-sum(sign*f>0 for f in base+donors),0) for sign in (1,-1))
    assert score["direction_floor_numerator"]==30*direct
    assert score["denominator"]==30*len(truth)
    proof=score["proof"]
    if proof["kind"]=="rational_dual":
        y=[0]*len(a)
        pairs=proof["nonzero_weights"]
        assert len(set(row for row,v in pairs))==len(pairs)
        s=proof["scale"]
        assert isinstance(s,int) and s>0
        for row,v in pairs:
            assert 0<=row<len(y) and abs(v)<=s
            y[row]=v
        bound=sum(v*b for v,b in zip(y,residual))
        for j in range(len(donors)):
            bound-=max(0,sum(v*col[j] for v,col in zip(y,a)))
        assert bound==proof["numerator"]
        cert=max(cell_floor,-((-bound)//s))
    else:
        assert proof["kind"]=="interval_floor"
        cert=cell_floor
    assert cert==score["integer_lower"] and score["integer_gap"]==witness-cert
    brute, tested=exhaustive(residual,a,donors)
    if brute is not None:
        assert cert<=brute<=witness
    if score["certified"]:
        assert cert==witness==score["optimal_numerator"]
        assert brute is None or brute==witness
    return dict(witness=witness,lower=cert,brute=brute,enums=tested,proof=proof["kind"])


def check():
    # Synthetic windows contain both a leading partial window and a trailing partial window.
    n=750
    base=[]; donors=[401]; truth=[1]
    r,a=cells(base,donors,truth,n)
    cost, tested=exhaustive(r,a,donors)
    assert cost==30 and tested==[2,1]
    assert sum(max(0,-v,v-sum(row)) for v,row in zip(r,a))==16
    print("Independent direct-cell and binary-enumeration checks passed")


def canonical_hash(record):
    data = (json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(data).hexdigest()


def verify_archive(path):
    bundle = json.loads(gzip.decompress(path.read_bytes()))
    geometry = bundle["geometry"]
    results = bundle["result"]
    if (bundle["study"] != 110 or canonical_hash(geometry) != bundle["geometry_sha256"]
            or canonical_hash(results) != bundle["result_sha256"]
            or results["geometry_sha256"] != bundle["geometry_sha256"]):
        raise ValueError("Evidence fingerprint mismatch")
    required = {f"{role}_{fold}_{seed}_{part}" for fold in (33,35) for seed in (0,3,6)
                for role in ("source","target") for part in (("head","tail") if role=="source" else ("head",))}
    checked = {}
    proofs = Counter()
    work = 0
    for cell in geometry["cases"]:
        case = f"{cell['role']}_{cell['fold']}_{cell['anchor_seed']}_{cell['part']}"
        if case in checked:
            raise ValueError(f"Duplicate case {case}")
        expected = results["cases"][case]
        oracle = expected["oracle"]
        if not oracle["certified"]:
            raise ValueError(f"Uncertified optimum in {case}")
        anchor = [r[1] for r in cell["kept"]]
        donors = [r["b"][1] for r in cell["donors"]]
        truth = bundle["evidence"]["truth"][cell["clip"]]
        witness_ids = [cell["donors"][i]["b"][0] for i in oracle["selected_donor_indices"]]
        if witness_ids != expected["selected_b_ids"]:
            raise ValueError(f"Witness ID mapping differs in {case}")
        report = verify(anchor, donors, truth, cell["nframes"], oracle)
        if report["brute"] is None:
            raise ValueError(f"{case} exceeds this study's 18 optional events per direction")
        checked[case] = report["brute"]
        proofs[report["proof"]] += 1
        work += sum(report["enums"])
    if set(checked) != required or set(results["cases"]) != required:
        raise ValueError("Expected all 18 registered cases")
    print(f"Verified {len(checked)} optima, {work:,} direction-specific subsets, and {dict(proofs)}.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    args=parser.parse_args()
    check()
    verify_archive(args.archive)
