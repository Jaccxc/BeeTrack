"""Independent verifier for §108. Checks masks, controls, window counts and maximum TP."""
import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
from scipy.optimize import linear_sum_assignment


FOLDS = (33, 35)
TOL = 1e-9


def require_equal(actual, expected, key):
    if isinstance(expected, dict):
        for name, value in expected.items():
            require_equal(actual[name], value, f"{key}/{name}")
    elif isinstance(expected, (np.ndarray, list, tuple, float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9, err_msg=key)
    elif actual != expected:
        raise AssertionError(f"{key}: {actual} != {expected}")


@lru_cache(maxsize=2)
def independent_lags(n):
    period = n // 25
    distances = np.arange(period)
    allowed = np.minimum(distances, period - distances) >= 6
    valid = allowed[:, None] & allowed[None, :] & allowed[(distances[:, None] - distances) % period]
    candidates = np.argwhere(valid)
    picked = np.random.Generator(np.random.PCG64(108)).choice(len(candidates), 99, replace=False)
    return candidates[picked]


def canonical(data):
    ids, signed, coords = [], [], []
    for row in sorted(data, key=lambda r: r[0]):
        i, t = row[:2]
        assert int(i) == i and int(t) == t and t != 0
        ids.append(int(i))
        signed.append(int(t))
        coords.append([int(round(value * 100)) for value in row[2:6]])
    assert len(set(ids)) == len(ids)
    return np.asarray(ids, int), np.asarray(signed, int), np.array(coords, np.int64).reshape(-1, 4)


def supports(anchor, buddy, n, lags):
    a_id, a_t, a_box = canonical(anchor)
    b_id, b_t, b_box = canonical(buddy)
    # Compute only the spatially admissible edge list. At each lag fill a full
    # sorted matrix so SciPy's prescribed tie behaviour remains reproducible.
    pairs = []
    b_area = (b_box[:, 2] - b_box[:, 0]) * (b_box[:, 3] - b_box[:, 1])
    for i, (x1, y1, x2, y2) in enumerate(a_box):
        dx = np.maximum(0, np.minimum(x2, b_box[:, 2]) - np.maximum(x1, b_box[:, 0]))
        dy = np.maximum(0, np.minimum(y2, b_box[:, 3]) - np.maximum(y1, b_box[:, 1]))
        area = (x2-x1) * (y2-y1)
        cut = 2 * dx * dy >= area + b_area - dx * dy
        for j in np.flatnonzero(cut & ((b_t > 0) == (a_t[i] > 0))):
            pairs.append((i, j))
    i, j = np.array(pairs, int).reshape(-1, 2).T
    anchor_f, buddy_f = np.abs(a_t[i]), np.abs(b_t[j])
    shape = (len(a_id), len(b_id))
    results = {}
    for lag in sorted(set(lags) | {0}):
        t = 1 + ((buddy_f + 25 * lag - 1) % n)
        good = np.abs(anchor_f - t) <= 5
        matrix = np.zeros(shape, dtype=bool)
        matrix[i[good], j[good]] = True
        left, right = linear_sum_assignment(matrix, maximize=True)
        keep = matrix[left, right]
        results[lag] = [(int(a_id[i]), int(b_id[j])) for i, j in zip(left[keep], right[keep])]
    return results


def check_mask(trio, seed, n, recorded):
    lags = independent_lags(n)
    anchor = trio[seed]
    friends = sorted(k for k in trio if k != seed)
    tables = [supports(anchor, trio[k], n, lags[:, c]) for c, k in enumerate(friends)]
    real_pairs = [table[0] for table in tables]
    retained = sorted({a for table in real_pairs for a, _ in table})
    phase_counts = [len({a for table, lag in zip(tables, cut) for a, _ in table[lag]}) for cut in lags]
    for c, pairs in enumerate(real_pairs):
        require_equal(recorded["pairs"][c], pairs, f"mask/{seed}/buddy/{c}")
    require_equal(recorded, {"retained": retained, "phase_counts": phase_counts, "phase_lags": lags}, f"mask/{seed}")
    return retained, phase_counts


class Counter:
    def __init__(self, n, truth):
        frames = np.arange(n)
        self.starts, self.ends = [], []
        for offset in range(30):
            changes = np.flatnonzero(np.diff((frames + 25 * offset) // 750)) + 1
            self.starts.extend([0, *changes])
            self.ends.extend([*changes, n])
        self.n = n
        self.gt = self.bins(truth)
        self.g_totals = self.gt.sum(axis=0) / 30

    def bins(self, signed):
        signs = np.array(signed, dtype=int)
        buckets = np.zeros((self.n + 1, 2), dtype=int)
        np.add.at(buckets, (np.abs(signs), (signs < 0).astype(int)), 1)
        prefix = buckets.cumsum(axis=0)
        return prefix[self.ends] - prefix[self.starts]

    def measure(self, bins):
        totals = bins.sum(axis=0) / 30
        delta = totals - self.g_totals
        components = np.sum(np.abs(bins - self.gt), axis=0) / (30 * sum(self.g_totals)) * 100
        return {"totals": totals.tolist(), "bias": delta.tolist(), "floor": sum(np.abs(delta))/sum(self.g_totals)*100,
                "window": sum(components), "window_by_direction": components.tolist()}


def maximum_hits(pred_ids, times, adjacency):
    # Unweighted bipartite maximum via independent alternating augmenting paths.
    owner = {}

    def extend(pid, visited):
        for gid in adjacency.get(pid, ()):
            if gid in visited:
                continue
            visited.add(gid)
            if gid not in owner or extend(owner[gid], visited):
                owner[gid] = pid
                return True
        return False

    for pid in pred_ids:
        extend(pid, set())
    plus = sum(times[pid] > 0 for pid in owner.values())
    return np.array([plus, len(owner) - plus])


def evaluate(candidates, times, counter, graphs):
    summary = counter.measure(counter.bins([times[i] for i in candidates]))
    nt = sum(counter.g_totals)
    for threshold, graph in graphs.items():
        hits = maximum_hits(candidates, times, graph)
        tp = int(sum(hits))
        summary[threshold] = {"tp": tp, "fp": len(candidates)-tp, "fn": nt-tp,
                              "f1": 2*tp/(nt+len(candidates)), "tp_by_direction": hits.tolist(),
                              "fp_by_direction": np.asarray(summary["totals"])-hits,
                              "fn_by_direction": counter.g_totals-hits}
    return summary


def score_case(frozen, answer, trio, seed, n, result):
    retained = frozen["retained"]
    times = {int(row[0]): int(row[1]) for row in trio[seed]}
    truth = {int(row[0]): int(row[1]) for row in answer["gt"]}
    counter = Counter(n, list(truth.values()))
    graphs = {}
    for tau in ("05", "03"):
        neighbors = defaultdict(list)
        for i, j in answer["edges"+tau]:
            assert (times[i] > 0) == (truth[j] > 0)
            neighbors[i].append(j)
        graphs[tau] = neighbors
    keys = sorted(times)
    base = evaluate(keys, times, counter, graphs)
    kept = evaluate(retained, times, counter, graphs)
    mean = sum(counter.bins([event[1] for event in rows]) for rows in trio.values()) / 3
    repeat_e3 = counter.measure(mean)
    require_equal(result["e3"], repeat_e3, "E3")
    if answer.get("group_refs"):
        require_equal(repeat_e3["window"], answer["group_refs"]["P0"], "published group E3")
    if answer.get("refs"):
        require_equal(round(base["window"], 2), answer["refs"]["P0"], "published P0")
    require_equal(result["full"], base, "full")
    require_equal(result["kept"], kept, "kept")
    r = len(keys) - len(retained)
    for tau in ("05", "03"):
        lost = base[tau]["tp"] - kept[tau]["tp"]
        x = lost * (len(truth) + len(keys)) - r * base[tau]["tp"]
        expected = dict(lost_tp=lost, lost_fp=r-lost, loss_fraction=lost/r if r else None,
                        f1_equal=x == 0, f1_improved=x < 0)
        require_equal(result["loss"+tau], expected, f"loss/{tau}")
    rng = np.random.Generator(np.random.PCG64(108))
    pools = [[k for k in keys if (times[k] > 0) == flag] for flag in (True, False)]
    budgets = [int(x) for x in kept["totals"]]
    assert len(result["quota"]) == 99
    quota = []
    for j, record in enumerate(result["quota"]):
        selection = sorted(k for pool, size in zip(pools, budgets) for k in rng.choice(pool, size, replace=False))
        require_equal(record["ids"], selection, f"quota/{j}/IDs")
        trial = evaluate(selection, times, counter, graphs)
        require_equal(record, trial, f"quota/{j}")
        quota.append(trial)
    med = {"window": median(q["window"] for q in quota)}
    for tau in ("05", "03"):
        med["tp"+tau] = int(median(q[tau]["tp"] for q in quota))
        med["f1"+tau] = median(q[tau]["f1"] for q in quota)
    phase = frozen["phase_counts"]
    wins = sum(len(retained) > x for x in phase)
    passed = r >= 10 and len(retained) >= 10 and wins >= 95 and kept["05"]["tp"] >= med["tp05"]+2
    require_equal(result, {"removed": r, "retained": len(retained), "gt_totals": counter.g_totals,
                           "quota_medians": med, "quota_median05_tp": med["tp05"], "phase_counts": phase,
                           "phase_wins": wins, "source_anchor_pass": passed}, "result")
    return passed


def component_values(rows):
    def effect(differences):
        return dict(gain=sum(differences)/len(differences), wins=sum(v > TOL for v in differences))
    return dict(window_vs_p0=effect([r["full"]["window"]-r["kept"]["window"] for r in rows]),
                window_vs_e3=effect([r["e3"]["window"]-r["kept"]["window"] for r in rows]),
                window_vs_quota=effect([r["quota_medians"]["window"]-r["kept"]["window"] for r in rows]),
                f1_vs_p0=effect([r["kept"]["05"]["f1"]-r["full"]["05"]["f1"] for r in rows]),
                f1_vs_quota=effect([r["kept"]["05"]["f1"]-r["quota_medians"]["f105"] for r in rows]))


def verify_folds(role, result, anchors):
    for fold in FOLDS:
        reported = result["folds"][str(fold)]
        if role == "source":
            for suffix in ("head", "tail"):
                flags = [anchors[f"{fold}_{s}_{suffix}"] for s in range(9)]
                votes = [sum(flags[i:i+3]) for i in (0, 3, 6)]
                require_equal(reported[suffix], dict(anchor_pass=flags, group_votes=votes,
                              group_pass=[v >= 2 for v in votes], passes=sum(v >= 2 for v in votes) >= 2), suffix)
        else:
            rows = [result["cases"][f"{fold}_{s}_head"] for s in range(9)]
            main = rows[::3]
            comp = component_values(main)
            f1, w = comp["f1_vs_p0"], comp["window_vs_e3"]
            window_pass = w["gain"] >= 2 and w["wins"] >= 2 and comp["window_vs_p0"]["gain"] > TOL
            wins = sum(r["kept"]["05"]["tp"] > r["quota_medians"]["tp05"] for r in main)
            f1_pass = f1["gain"] >= .02 and f1["wins"] >= 2 and wins >= 2
            require_equal(reported["primary"], dict(comp, tp_wins_vs_quota=wins, window_pass=window_pass,
                                                    f1_pass=f1_pass), "target primary")
            require_equal(reported["passes"], window_pass and f1_pass, "target gate")
            for i in range(3):
                require_equal(reported["rotations"][i], component_values(rows[3*i:3*i+3]), f"rotations/{i}")
    if role == "source" and "advance" in result:
        require_equal(result["advance"], any(f["head"]["passes"] for f in result["folds"].values()), "advance")


def quota_ids(anchor, retained):
    by_dir = [[int(row[0]) for row in sorted(anchor) if (row[1]>0) == positive] for positive in (True, False)]
    quotas = [len(set(pool).intersection(retained)) for pool in by_dir]
    rng = np.random.Generator(np.random.PCG64(108))
    return [sorted(k for pool,count in zip(by_dir, quotas) for k in rng.choice(pool, count, replace=False))
            for _ in range(99)]


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data), hashlib.sha256(data).hexdigest()


def digest_record(obj):
    data = (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(data).hexdigest()


def walk(masks, labels=None, results=None, do_masks=True):
    predictions, controls, role = masks["predictions"], masks["controls"], masks["role"]
    parts = ("head", "tail") if role == "source" else ("head",)
    expected = {f"{fold}_{seed}_{part}" for fold in FOLDS for seed in range(9) for part in parts}
    if role not in ("source", "target") or set(predictions) != expected or set(controls) != expected:
        raise ValueError("Unexpected or absent fold, seed, or partition")
    if labels is not None and (set(labels) != expected or set(results["cases"]) != expected):
        raise ValueError("Unexpected or absent score rows")
    flags = {}
    for fold in FOLDS:
        for part in parts:
            for seed in range(9):
                name = f"{fold}_{seed}_{part}"
                group = range(3*(seed//3), 3*(seed//3)+3)
                trio = {j: predictions[f"{fold}_{j}_{part}"]["predictions"] for j in group}
                n = 3950 if part == "head" else 1000
                row = controls[name]
                if do_masks:
                    kept, _ = check_mask(trio, seed, n, row)
                    require_equal(row["quota_masks"], quota_ids(trio[seed], kept), f"quota freeze/{name}")
                if labels is not None:
                    flags[name] = score_case(row, labels[name], trio, seed, n, results["cases"][name])
            print(f"Checked {role}, fold {fold}, {part}: 9 anchor seeds", flush=True)
    if labels is not None:
        verify_folds(role, results, flags)


def report_table(report, labels):
    print("fold part seed kept/total phase_wins/99  TP05_kept/med_quota  F1_05_P0/kept  "
          "F1_03_P0/kept  window_P0/kept/E3/med_quota  signed_bias_kept_in/out  floor_kept")
    for key in sorted(report["cases"], key=lambda x: (int(x.split("_")[0]), x.split("_")[2], int(x.split("_")[1]))):
        fold, s, part = key.split("_")
        row = report["cases"][key]
        base, kept, q = row["full"], row["kept"], row["quota_medians"]
        print(f"{fold} {part} {s} {row['retained']}/{row['removed']+row['retained']} {row['phase_wins']} "
              f"{kept['05']['tp']}/{q['tp05']} {base['05']['f1']:.3f}/{kept['05']['f1']:.3f} "
              f"{base['03']['f1']:.3f}/{kept['03']['f1']:.3f} "
              f"{base['window']:.2f}/{kept['window']:.2f}/{row['e3']['window']:.2f}/{q['window']:.2f} "
              f"{kept['bias'][0]:+g}/{kept['bias'][1]:+g} {kept['floor']:.2f}")
    if any(row["refs"] for row in labels.values()):
        print("fold seed  P0  kept  ByteTrack  calibrated_ByteTrack  source_selected_TOPICTrack")
        for key in sorted(labels):
            row, label = report["cases"][key], labels[key]
            refs = label["refs"]
            topic = refs.get("TOPICTrack selected")
            print(label["fold"], label["seed"], f"{row['full']['window']:.2f}", f"{row['kept']['window']:.2f}",
                  refs["ByteTrack"], refs["calibrated ByteTrack"], "-" if topic is None else topic)
        print("fold anchor  kept  E3_P0  E3_ByteTrack  E3_calibrated_ByteTrack")
        for key in sorted(labels):
            label = labels[key]
            if label["seed"] in (0,3,6):
                row, ref = report["cases"][key], label["group_refs"]
                print(label["fold"], label["seed"], f"{row['kept']['window']:.2f}",
                      *(f"{ref[name]:.2f}" for name in ("P0", "ByteTrack", "calibrated ByteTrack")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("masks", "scores"))
    parser.add_argument("input", type=Path, help="frozen mask JSON, or the public replay archive for scores")
    parser.add_argument("--report", type=Path, help="compare against the original full result JSON")
    parser.add_argument("--table", action="store_true", help="print per-seed results and controls after checking")
    parser.add_argument("--already-checked-masks", action="store_true",
                        help="score only after this mask manifest has separately passed the masks stage")
    args = parser.parse_args()
    data, hashval = read(args.input)
    if args.stage == "masks":
        if args.already_checked_masks or args.report:
            parser.error("masks stage always performs the complete GT-free mask and draw check")
        walk(data)
    else:
        masks = data["masks"]
        if digest_record(masks) != data["masks_sha256"]:
            raise ValueError("Frozen masks fingerprint mismatch")
        if args.report:
            report, report_hash = read(args.report)
            if report_hash != data["result_sha256"] or report["masks_sha256"] != data["masks_sha256"]:
                raise ValueError("Full results fingerprint mismatch")
        else:
            # The public archive includes frozen case digests, not repetitive draw scores.
            import run108
            cases, folds = run108.score_rows(masks, data["labels"])
            report = dict(cases=cases, folds=folds)
        if ({key: digest_record(row) for key,row in report["cases"].items()} != data["case_sha256"]
                or report["folds"] != data["folds"]):
            raise ValueError("Case fingerprints or fold decisions differ from the archive")
        walk(masks, data["labels"], report, not args.already_checked_masks)
        if args.table:
            report_table(report, data["labels"])
    print(f"PASS {args.stage}, input SHA-256 {hashval}")


if __name__ == "__main__":
    main()
