"""Independently replay §103 from its signed event frames and matched counts.

Uses direct event-to-window binning and exhaustive disjoint switch selections.
Box-level candidate discovery and frame-level event matching require the cached
MOT inputs (see run_study.py). Only NumPy is needed for this archive replay.
"""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def check_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def bins(stream, nframes, shift=0):
    """Group each signed, 1-based event into all 30 predeclared window grids."""
    stream = np.asarray(stream, dtype=np.int64)
    assert np.all((1 <= np.abs(stream)) & (np.abs(stream) <= nframes))
    nbin = (nframes + 725 + 749) // 750
    frame = (np.abs(stream) - 1 + shift) % nframes
    offsets = np.arange(30)[:, None]
    cells = 2 * (offsets * nbin + (frame + 25 * offsets) // 750) + (stream < 0)
    return np.bincount(cells.ravel(), minlength=60 * nbin).reshape(-1, 2)


def error(pred, truth):
    return float(100 * np.abs(pred - truth).sum() / truth.sum())


def verify_phase(truth, stream, nframes, scale, saved):
    aligned = error(bins(stream, nframes) * scale, truth)
    shifts = np.array([error(bins(stream, nframes, shift) * scale, truth)
                       for shift in range(25, nframes, 25)])
    q10, median, q90 = np.quantile(shifts, (0.1, 0.5, 0.9))
    fraction_le = np.mean((shifts < aligned) | np.isclose(shifts, aligned, rtol=1e-12, atol=1e-12))
    expected = dict(original=aligned, shifted_median=median, gain_pp=median-aligned, shifted_q10=q10,
                    shifted_q90=q90, shifted_sd_population=shifts.std(), shifted_sd_sample=shifts.std(ddof=1),
                    fraction_le=fraction_le, n_shifts=len(shifts))
    assert saved.keys() == expected.keys() and len(shifts) == 157
    for key, value in expected.items():
        check_close(value, saved[key])
    return expected


def disjoint(pairs, chosen):
    ids = [tid for i in chosen for tid in pairs[i][:2]]
    return len(ids) == len(set(ids))


def verify_swaps(row, truth, nframes):
    base = bins(row["events"], nframes)
    pairs, deltas, kinds = [], [], Counter()
    for candidate in row["candidates"]:
        left, right, cut = candidate["pair"]
        assert left < right and 3 <= cut <= nframes - 3
        assert not (Counter(candidate["before"]) - Counter(row["events"]))
        delta = bins(candidate["after"], nframes) - bins(candidate["before"], nframes)
        total = delta.sum(axis=0)
        assert total[0] == total[1]
        kind = "changed_totals" if total.any() else "timing_only" if delta.any() else "same_windows"
        assert kind == candidate["kind"]
        kinds[kind] += 1
        pairs.append((left, right, cut))
        deltas.append(delta)
    census = row["census"]
    assert len(pairs) == len(set(pairs)) == census["close_pairs"] <= census["reciprocal_pairs"]
    for kind in ("changed_totals", "timing_only", "same_windows"):
        assert census[kind] == kinds[kind]
    earliest = []
    for i in sorted(range(len(pairs)), key=lambda i: (pairs[i][2], pairs[i][0], pairs[i][1])):
        if disjoint(pairs, earliest + [i]):
            earliest.append(i)
    optimal = min(error(base + sum(deltas[i] for i in idx), truth)
                  for k in range(len(pairs) + 1) for idx in combinations(range(len(pairs)), k)
                  if disjoint(pairs, idx))
    arms = row["arms"]
    assert arms.keys() == {"p0", "chronological", "oracle"}
    assert arms["p0"]["positions"] == [] and arms["chronological"]["positions"] == earliest
    check_close(optimal, arms["oracle"]["window"])
    for arm in arms.values():
        chosen = arm["positions"]
        assert all(0 <= i < len(pairs) for i in chosen) and disjoint(pairs, chosen)
        counts = base + sum(deltas[i] for i in chosen)
        assert (counts >= 0).all()
        check_close(error(counts, truth), arm["window"])
        check_close((counts - truth).sum(axis=0) / 30, [arm["bias_in"], arm["bias_out"]])
        assert arm["scores"].keys() == {"0.3", "0.5"}
        for score in arm["scores"].values():
            tp, fp, fn = (score[key] for key in ("tp", "fp", "fn"))
            assert min(tp, fp, fn) >= 0
            assert 30 * (tp + fp) == counts.sum() and 30 * (tp + fn) == truth.sum()
            check_close(2 * tp / (2 * tp + fp + fn), score["f1"])
    return error(base, truth) - optimal


def replay(data, table=False):
    if not __debug__:
        raise RuntimeError("Run with assertions enabled to verify the archive.")
    assert data["study"] == 103 and set(data["clips"]) == {"33", "35"}
    cases = {(r["hive"], r["seed"], r["method"]): r for r in data["cases"]}
    expected = {(h, s, m) for h in (33, 35) for s in range(9)
                for m in ("P0", "ByteTrack", "calibrated ByteTrack", "TOPICTrack selected")
                if not (m == "TOPICTrack selected" and s >= 3)}
    assert len(data["cases"]) == 60 and cases.keys() == expected
    for hive in (33, 35):
        clip = data["clips"][str(hive)]
        nframes = clip["frames"]
        assert nframes == 3950
        truth = bins(clip["gt"], nframes)
        assert truth.sum() == 30 * len(clip["gt"])
        ceiling, timing, passed, total = [], [], 0, 0
        for (h, seed, method), row in sorted(cases.items()):
            if h != hive:
                continue
            if method == "calibrated ByteTrack":
                stream = cases[h, seed, "ByteTrack"]["events"]
                scale = np.array([n / d for n, d in row["scale_ratio"]])
            else:
                stream, scale = row["events"], 1
            stats = verify_phase(truth, stream, nframes, scale, row["phase"])
            if method == "P0":
                ceiling.append(verify_swaps(row, truth, nframes))
                passed += stats["fraction_le"] <= 0.10
                total += len(row["candidates"])
                timing.append(stats["gain_pp"])
            else:
                assert round(stats["original"], 2) == row["reference_error"]
            if table:
                print(f"{h} {seed} {method:21s} original={stats['original']:.2f} "
                      f"shifted={stats['shifted_median']:.2f} fraction_le={stats['fraction_le']:.3f}")
        print(f"hive {hive}: {total} eligible reciprocal swaps across 9 seeds; mean oracle ceiling "
              f"{np.mean(ceiling):.3f} pp; median timing gain {np.median(timing):.2f} pp; "
              f"{passed}/9 seeds at or below the 10% shifted-error rank.")
    print("Verified all 18 P0 cases, 42 comparison cases and 157 shifted errors per case against stored summaries.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[2] / "docs/study103_results.json")
    p.add_argument("--table", action="store_true")
    args = p.parse_args()
    replay(json.loads(args.input.read_text(encoding="utf-8")), args.table)
