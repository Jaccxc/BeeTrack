"""Verify §111 from portable case evidence, or compare against the original official matcher.

Run from the repository root. The optional --mots directory contains native.txt and
nearall_B0.txt, etc. --mots uses only dev head 33 or 35 and requires --hive explicitly.
"""

import argparse
from fractions import Fraction
import gzip
import hashlib
from itertools import product
import json
import os
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def digest(path):
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def rank(graph):
    if not graph.shape[0] or not graph.shape[1] or not graph.nnz:
        return 0
    return int(np.count_nonzero(maximum_bipartite_matching(graph, perm_type="column") >= 0))


def extrema(graph, selected, total):
    """Extend a maximum matching on S by alternating augmentation to global rank M.

    Every already matched GT remains matched during augmentation, so max selected
    TP is rank(S). Maximizing matches in the complement attains M-rank(complement).
    """
    return total - rank(graph[:, ~selected]), rank(graph[:, selected])


def official_events(ref, rows, polygon):
    ordered = ref.by_track(rows)
    tracks = ref.zone_tracks(ordered, polygon)
    times = {}
    for tid, (direction, _, _, crossed) in tracks.items():
        if direction:
            j = next(j for j, side in reversed(crossed) if side == direction)
            times[int(tid)] = int(ordered[j, 0]) * (1 if direction == "in" else -1)
    return times


def capture_official(ref, gt, mot, polygon, thr, tids, pids):
    captured = []

    def original(matrix, *args, **kwargs):
        assignment = linear_sum_assignment(matrix, *args, **kwargs)
        if kwargs.get("maximize", False):
            captured.append(matrix)
        return assignment

    with patch.object(ref, "linear_sum_assignment", original), patch.object(
        ref.frame_match, "__defaults__", (thr,)
    ):
        score = ref.match_events(gt, mot, polygon)
    assert len(captured) == 1
    graph = captured[0]
    assert graph.shape == (len(pids), len(tids))
    assert rank(csr_matrix(graph)) == score["tp"]
    return {(pids[i], tids[j]) for i, j in zip(*np.nonzero(graph))}


def check_mots(package, mots, hive):
    from beetrack import evaluate as ref
    from beetrack.common import RUNS
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    evidence = package["evidence"]
    assert hive in (33, 35) and evidence["nframes"] == 3950
    root = mots.parent.parent
    complete = read(root / "tracks_done.json")
    assert set(complete["cases"]) == {f"{h}_{s}" for h in (33, 35) for s in range(9)}
    marker = mots / "done.json"
    assert digest(marker) == complete["cases"][mots.name]
    cell = read(marker)
    frozen = root / "target" / (mots.name+".json")
    assert cell["freeze_sha256"] == complete["freeze_sha256"] == digest(root/"freeze111.json")
    assert cell["case_sha256"] == digest(frozen)
    assert (hive, cell["seed"]) == check_frozen(package, frozen)
    assert cell["arms"].keys() == evidence["arms"].keys() and cell["fold"] == hive
    native = package["results"]["arms"]["native"]
    reference = [round(native["window"], 2), sum(native["bias"]),
                 *[native["05"][name] for name in ("tp", "fp", "fn")]]
    assert reference == cell["native_p0_reference"], (hive, cell["seed"], "native versus §95")
    seq = f"BEE24-{hive}-head"
    gt = ref.load_tracks(RUNS / "gt" / "dev" / seq / "gt" / "gt.txt")
    polygon = read(ref.ZONES)[seq]
    truth = official_events(ref, gt, polygon)
    assert truth == dict(evidence["gt"])
    for name, item in evidence["arms"].items():
        path = mots / f"{name}.txt"
        assert cell["arms"][name] == digest(path)
        mot = ref.load_tracks(path)
        times = official_events(ref, mot, polygon)
        assert times == dict(item["predictions"]), (name, "event times")
        for suffix, threshold in (("05", .5), ("03", .3)):
            edges = capture_official(ref, gt, mot, polygon, threshold, sorted(truth), sorted(times))
            assert edges == set(map(tuple, item["edges" + suffix])), (name, suffix, "legal pairs")


def direct_windows(truth, pred, nframes):
    # Intersect every half-open window with the clip, retaining partial windows at either end.
    assert all(1 <= abs(t) <= nframes for t in (*truth, *pred))
    numerators = [0, 0]
    for lead in range(0, 750, 25):
        for start in range(1-lead, nframes+1, 750):
            end = start+750
            for j, sign in enumerate((1, -1)):
                ng = sum(start <= sign*t < end and sign*t > 0 for t in truth)
                npred = sum(start <= sign*t < end and sign*t > 0 for t in pred)
                numerators[j] += abs(ng-npred)
    return numerators


def check_arm(item, recorded, truth, cohorts, nframes):
    pred = dict(item["predictions"])
    assert len(pred) == len(item["predictions"])
    gt_ids, pred_ids = sorted(truth), sorted(pred)
    gt_idx = {g: i for i, g in enumerate(gt_ids)}
    pred_idx = {p: i for i, p in enumerate(pred_ids)}
    gt_counts = [sum(t > 0 for t in truth.values()), sum(t < 0 for t in truth.values())]
    pr_counts = [sum(t > 0 for t in pred.values()), sum(t < 0 for t in pred.values())]
    assert pr_counts == recorded["pred_totals"] and gt_counts == recorded["gt_totals"]
    assert [p-g for p, g in zip(pr_counts, gt_counts)] == recorded["bias"]
    num = direct_windows(truth.values(), pred.values(), nframes)
    den = 30*len(truth)
    assert num == recorded["direction_window_numerators"]
    assert (sum(num), den) == (recorded["window_numerator"], recorded["window_denominator"])
    assert abs(100*sum(num)/den - recorded["window"]) < 1e-10
    recomputed = {}
    for threshold in ("05", "03"):
        row = recorded[threshold]
        pairs = item["edges"+threshold]
        assert len(pairs) == len(set(map(tuple, pairs)))
        assert all((pred[p] > 0) == (truth[g] > 0) for p, g in pairs)
        ij = [(pred_idx[p], gt_idx[g]) for p, g in pairs]
        i, j = zip(*ij) if ij else ((), ())
        graph = csr_matrix((np.ones(len(ij), dtype=np.int8), (i, j)), shape=(len(pred), len(truth)))
        total = rank(graph)
        assert [row[k] for k in ("tp", "fp", "fn")] == [total, len(pred)-total, len(truth)-total]
        assert abs(row["f1"] - 2*total/(len(truth)+len(pred))) < 1e-12
        direction = np.array([truth[g] > 0 for g in gt_ids])
        assert [rank(graph[:, direction]), rank(graph[:, ~direction])] == row["tp_by_direction"]
        # Only the canonical, descriptive value reuses the official sorted SciPy tie policy.
        dense = graph.toarray()
        pi, gi = linear_sum_assignment(dense, maximize=True)
        matched = {gt_ids[c] for c in gi[dense[pi, gi] > 0]}
        indegree = {g for _, g in pairs}
        recomputed[threshold] = {}
        for cohort, ids in cohorts.items():
            chosen = set(ids)
            assert chosen <= truth.keys() and len(ids) == len(chosen)
            mask = np.array([g in chosen for g in gt_ids])
            lo, hi = extrema(graph, mask, total)
            fixed = len(chosen & matched)
            b = row["target_bounds"][cohort]
            assert (len(chosen), lo, hi, fixed) == tuple(b[k] for k in ("n", "low", "high", "canonical"))
            assert sorted(chosen-indegree) == b["always_unmatched_ids"]
            recomputed[threshold][cohort] = (lo, hi, fixed)
    return recomputed


def phase(bounds, cohort, prefix, threshold):
    b0, b1, s0, s1 = [bounds[prefix+"_"+p][threshold][cohort] for p in ("B0", "B1", "S0", "S1")]
    return (s0[0]+s1[0]-b0[1]-b1[1], s0[1]+s1[1]-b0[0]-b1[0],
            s0[2]+s1[2]-b0[2]-b1[2])


def check_case(package):
    e, r = package["evidence"], package["results"]
    truth, cohorts = dict(e["gt"]), e["target_ids"]
    assert truth and len(truth) == len(e["gt"])
    assert e["arms"].keys() == r["arms"].keys() and "native" in e["arms"]
    bounds = {k: check_arm(v, r["arms"][k], truth, cohorts, e["nframes"]) for k, v in e["arms"].items()}
    expected = {}
    for threshold in ("05", "03"):
        for prefix, cohort in (("nearall", "all"), ("near", "paired"), ("far", "paired")):
            present = {p for p in ("B0", "B1", "S0", "S1") if prefix + "_" + p in bounds}
            assert not present or len(present) == 4
            if present:
                expected.setdefault(prefix, {})[threshold] = phase(bounds, cohort, prefix, threshold)
        if "near" in expected or "far" in expected:
            nl, nu, nc = expected["near"][threshold]
            fl, fu, fc = expected["far"][threshold]
            expected.setdefault("near_minus_far", {})[threshold] = (nl-fu, nu-fl, nc-fc)
    assert expected.keys() == r["contrasts"].keys()
    for label, both in expected.items():
        n = len(cohorts["all" if label == "nearall" else "paired"])
        for threshold, sums in both.items():
            rec = r["contrasts"][label][threshold]
            assert rec["n"] == n and rec["denominator"] == 2*n
            for k, value in zip(("lower", "upper", "canonical"), sums):
                assert rec[k+"_numerator"] == value
                assert rec[k] == (value/(2*n) if n else None)
    return expected


def check_frozen(package, path):
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data["study"] == 111 and data["role"] == "target" and data["fold"] in (33, 35)
    assert data["clip"] == f"BEE24-{data['fold']}-head" and data["nframes"] == 3950
    if "freeze_sha256" in package:
        assert hashlib.sha256(raw).hexdigest() == package["freeze_sha256"]
    groups = package["evidence"]["target_ids"]
    for label, field in (("all", "rows_all"), ("paired", "rows"), ("weak", "natural_weak_rows")):
        ids = sorted(row["gt_id"] for row in data[field])
        assert len(ids) == len(set(ids)) and ids == sorted(groups[label]), (label, "wrong frozen cohort")
    return data["fold"], data["seed"]


def aggregate(packages):
    """Use the equal-weight mean of *all* N>=10 seeds; counts are not pooled across a hive."""
    assert set(packages) == set(range(9))
    comparisons = {s: check_case(p) for s, p in packages.items()}
    paired_enabled = sum(len(p["evidence"]["target_ids"]["paired"]) >= 10 for p in packages.values()) >= 7
    for seed, p in packages.items():
        ids = p["evidence"]["target_ids"]
        assert not ids["all"] or "nearall" in comparisons[seed], (seed, "missing nearall treatment")
        if paired_enabled and ids["paired"]:
            assert "near_minus_far" in comparisons[seed], (seed, "missing paired treatment")
        if not paired_enabled:
            assert "near_minus_far" not in comparisons[seed], (seed, "unexpected paired treatment")
    nweak = [len(p["results"]["arms"]["native"]["05"]["target_bounds"]["weak"]["always_unmatched_ids"])
             for p in packages.values()]
    ngt = {len(p["evidence"]["gt"]) for p in packages.values()}
    assert len(ngt) == 1
    gt = ngt.pop()
    out = {"weak_median_count": median(nweak), "weak_pass": 100*median(nweak) >= 3*gt}
    for k, label in (("all", "nearall"), ("paired", "near_minus_far")):
        qualified = [s for s, p in packages.items() if len(p["evidence"]["target_ids"][k]) >= 10]
        rows = []
        for s in qualified:
            if label in comparisons[s]:
                n = len(packages[s]["evidence"]["target_ids"][k])
                rows.append([Fraction(i, 2*n) for i in comparisons[s][label]["05"][:2]])
        assert len(qualified) < 7 or len(qualified) == len(rows)
        lower = mean(r[0] for r in rows) if rows else None
        upper = mean(r[1] for r in rows) if rows else None
        positive = sum(r[0] > Fraction(1, 10**9) for r in rows)
        out[label] = dict(qualified=qualified, scored=len(rows), positive=positive,
                          mean_lower=str(lower) if rows else None, mean_upper=str(upper) if rows else None,
                          coverage=len(qualified) >= 7,
                          gate=positive >= 7 and lower >= Fraction(1, 10),
                          reverse=bool(rows) and upper <= Fraction(-1, 10))
    out["weak_link_gate"] = bool(out["nearall"]["gate"] and out["near_minus_far"]["gate"] and out["weak_pass"])
    return out


def self_check():
    # Enumerate the possible matched GT sets with optional unmatched prediction rows.
    checked = 0
    for npred, ngt in ((0, 2), (2, 0), (2, 3), (3, 2), (3, 3)):
        for bits in product((0, 1), repeat=npred*ngt):
            arr = np.array(bits, dtype=np.int8).reshape(npred, ngt)
            choices = [[-1, *np.flatnonzero(row)] for row in arr]
            hits = [set(p)-{-1} for p in product(*choices) if len(set(p)-{-1}) == sum(x >= 0 for x in p)]
            m = max(map(len, hits))
            graph = csr_matrix(arr)
            assert rank(graph) == m
            for membership in product((False, True), repeat=ngt):
                selected = set(np.flatnonzero(membership))
                values = {len(t & selected) for t in hits if len(t) == m}
                lo, hi = extrema(graph, np.array(membership, dtype=bool), m)
                assert values == set(range(lo, hi+1))
                checked += 1
    assert direct_windows([750, -1], [751, -1], 751) == [2, 0]
    assert direct_windows([1], [-1], 1) == [30, 30]
    print("Verified rank extrema for", checked, "synthetic graph/cohort pairs and window boundaries.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("self-check")
    case = sub.add_parser("case")
    case.add_argument("package", type=Path)
    case.add_argument("--mots", type=Path)
    case.add_argument("--frozen", type=Path, help="Independent frozen-selection case JSON.")
    case.add_argument("--hive", type=int, choices=(33, 35))
    fold = sub.add_parser("fold")
    fold.add_argument("packages", type=Path, nargs=9, help="Nine case packages in seed 0–8 order.")
    args = parser.parse_args()
    if args.cmd == "self-check":
        self_check()
    elif args.cmd == "case":
        p = read(args.package)
        if args.frozen:
            hive, _ = check_frozen(p, args.frozen)
            if args.hive and args.hive != hive:
                parser.error("--hive differs from frozen selection")
            args.hive = hive
        if args.mots:
            if args.hive is None:
                parser.error("--mots requires --hive")
            check_mots(p, args.mots, args.hive)
        check_case(p)
        print("Verified", len(p["evidence"]["arms"]), "arms at both IoU thresholds.")
    else:
        print(json.dumps(aggregate({s: read(p) for s, p in enumerate(args.packages)}), indent=2))


if __name__ == "__main__":
    main()
