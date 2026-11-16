"""Independent, CPU-only scorer for Study 126 detector-input conditions on dev heads.

Manifest format is the portable generator's schema 126 `cells` list. Each cell
has fold, seed, seq, family, arm, and manifest-relative MOT/archive paths with
SHA-256 digests. Run from the repo or pass --repo; --out and --prereg are required.
Only exact canonical dev head names are accepted. Results are written as one JSON
record per condition/case plus a paired summary; no detector data is loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

def find_repo(override: Path | None = None) -> Path:
    if override is not None:
        root = override.resolve()
        if not (root / "beetrack").is_dir():
            raise ValueError(f"--repo is not a BeeTrack checkout: {root}")
        return root
    starts = (Path.cwd().resolve(), Path(__file__).resolve())
    for start in starts:
        for parent in (start, *start.parents):
            if (parent / "beetrack").is_dir() and (parent / "tools").is_dir():
                return parent
            candidate = parent / "final_project"
            if (candidate / "beetrack").is_dir() and (candidate / "tools").is_dir():
                return candidate
    raise RuntimeError("Could not find BeeTrack repo from cwd or script path; pass --repo")


REPO = find_repo(Path(sys.argv[sys.argv.index("--repo") + 1]) if "--repo" in sys.argv else None)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.study117.physical117 import assert_score, verify_mot  # noqa: E402
from tools.study117.tracking.score117 import score_case  # noqa: E402

DEV = {"BEE24-33-head": ("BEE24-33", 1, 3950),
       "BEE24-35-head": ("BEE24-35", 1, 3950)}
HEADS = ("BEE24-33-head", "BEE24-35-head")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_mot(path: Path) -> tuple[bytes, np.ndarray]:
    raw = path.read_bytes()
    if not raw.strip():
        return raw, np.empty((0, 6), dtype=float)
    return raw, np.loadtxt(BytesIO(raw), delimiter=",", usecols=range(6), ndmin=2)


def score_one(cell: dict, mot_path: Path, prereg: str, runs: Path, zones: dict) -> dict:
    ev = cell["seq"]
    if ev not in HEADS or ev not in DEV:
        raise ValueError(f"Not an allowed dev head: {ev}")
    _, first, last = DEV[ev]
    nframes = last - first + 1
    gt_path = runs / "gt" / "dev" / ev / "gt" / "gt.txt"
    seqinfo = (runs / "gt" / "dev" / ev / "seqinfo.ini").read_text(encoding="utf-8")
    if f"seqLength={nframes}" not in seqinfo:
        raise ValueError(f"Prepared GT clock does not match registered {ev} length")
    gt = np.loadtxt(gt_path, delimiter=",", usecols=range(6), ndmin=2)
    if len(gt) and (gt[:, 0].min() < 1 or gt[:, 0].max() > nframes):
        raise ValueError(f"Prepared GT falls outside {ev} local clock")
    raw_mot, mot = load_mot(mot_path)
    if len(mot) and (mot[:, 0].min() < 1 or mot[:, 0].max() > nframes):
        raise ValueError(f"MOT contains out-of-range frames for {ev}")
    polygon = zones[ev]
    result = score_case(gt, mot, polygon, nframes)
    result["mot_sha256"] = sha(mot_path)
    independent = verify_mot(raw_mot, gt, polygon, nframes)
    assert_score(independent, result)
    evd = result["evidence"]
    gtime = {int(i): int(t) for i, t in evd["gt"]}
    ptime = {int(row[0]): int(row[1]) for row in evd["predictions"]}
    matched = {}
    for tau, s in result["iou"].items():
        pairs = s["matching"]
        errors = [abs(abs(ptime[p]) - abs(gtime[g])) for p, g in pairs]
        matched[tau] = dict(count=len(pairs), timing_abs_frames=errors,
                            timing_mean_frames=float(np.mean(errors)) if errors else None,
                            timing_median_frames=float(np.median(errors)) if errors else None,
                            timing_p90_frames=float(np.percentile(errors, 90)) if errors else None,
                            prediction_to_gt=pairs)
    win = result["window"]
    return dict(condition=f"{cell['family']}:{cell['arm']}", fold=str(cell["fold"]),
                seed=cell.get("seed"), family=cell["family"], arm=cell["arm"],
                case=ev, nframes=nframes, gt_rows=int(len(gt)),
                gt_events=int(len(evd["gt"])), gt_in=result["gt_counts"]["in"],
                gt_out=result["gt_counts"]["out"], pred_in=result["pred_counts"]["in"],
                pred_out=result["pred_counts"]["out"],
                window=dict(width_frames=750, offsets_frames=win["offsets_frames"],
                            gt_denominator=win["denominator"],
                            absolute_count_error_by_offset_in_out=win["l1_by_offset_in_out"],
                            integer_absolute_error_total=win["numerator"],
                            w30_percent=result["w30_percent"]),
                official_iou={tau: dict(tp=s["tp"], fp=s["fp"], fn=s["fn"], f1=s["f1"],
                                        by_direction=s["by_direction"], matched_bees=matched[tau])
                              for tau, s in result["iou"].items()},
                graph_evidence=evd,
                provenance=dict(mot_sha256=sha(mot_path),
                                gt_sha256=hashlib.sha256(np.ascontiguousarray(gt, dtype="<f8").tobytes()).hexdigest(),
                                prepared_gt=ev,
                                prereg=prereg, scorer="tools/study117/tracking/score117.py",
                                evidence_sha256=result["evidence_sha256"]))


def summarize(records: list[dict], gate_ready: bool) -> dict:
    by_case: dict[str, dict[str, dict[int | None, dict]]] = {}
    for r in records:
        by_case.setdefault(r["case"], {}).setdefault(r["condition"], {})[r["seed"]] = r
    conditions = sorted({r["condition"] for r in records})
    out = {}
    for cond in conditions:
        per = {}
        for ev, arms in sorted(by_case.items()):
            # Witness IDs define the untouched P0 detector-input arm as real:U.
            baseline = "real:U"
            if cond not in arms or baseline not in arms:
                continue
            p0s, bxs = arms[baseline], arms[cond]
            shared = sorted(set(p0s) & set(bxs), key=lambda x: -1 if x is None else x)
            if not shared:
                continue
            improvements, p0_w, bx_w = [], [], []
            f1drops = []
            for seed in shared:
                a, b = p0s[seed], bxs[seed]
                p0err = a["window"]["absolute_count_error_by_offset_in_out"]
                cderr = b["window"]["absolute_count_error_by_offset_in_out"]
                if len(p0err) != 30 or len(cderr) != 30:
                    raise ValueError("Expected exactly 30 phase offsets per head")
                improvements.append(a["window"]["w30_percent"] - b["window"]["w30_percent"])
                p0_w.append(a["window"]["w30_percent"])
                bx_w.append(b["window"]["w30_percent"])
                f1drops.append(b["official_iou"]["0.5"]["f1"] - a["official_iou"]["0.5"]["f1"])
            t_w = [arms["real:T"][s]["window"]["w30_percent"]
                   for s in shared if "real:T" in arms and s in arms["real:T"]]
            p0_range = max(p0_w) - min(p0_w) if len(p0_w) > 1 else 0.0
            t_range = max(t_w) - min(t_w) if len(t_w) > 1 else None
            gate_cut = max(p0_range, t_range) if t_range is not None else None
            f1_mean_diff = float(np.mean(f1drops))
            k_w = [arms["real:K"][s]["window"]["w30_percent"]
                   for s in shared if "real:K" in arms and s in arms["real:K"]]
            per[ev] = dict(seeds=shared, p0_w30_percent_by_seed=p0_w,
                           condition_w30_percent_by_seed=bx_w,
                           mean_w_improvement_percentage_points=float(np.mean(improvements)),
                           p0_seed_range=p0_range, t_seed_range=t_range,
                           larger_p0_t_seed_range=gate_cut,
                           greater_than_p0_t_range=(float(np.mean(improvements)) > gate_cut
                                                    if gate_ready and gate_cut is not None else None),
                           mean_f1_05_difference=f1_mean_diff,
                           f1_05_non_worse=(f1_mean_diff >= -1e-12 if gate_ready else None),
                           condition_f1_05_minus_p0_by_seed=f1drops,
                           k_mean_w_gt_t_mean_w=(float(np.mean(k_w)) > float(np.mean(t_w))
                                                if len(k_w) == len(t_w) and k_w else None),
                           per_seed_integer_window_errors={str(s): {
                               "p0": p0s[s]["window"]["integer_absolute_error_total"],
                               "condition": bxs[s]["window"]["integer_absolute_error_total"]}
                               for s in shared})
        out[cond] = per
    return out


def relative_artifact(manifest_path: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        raise ValueError(f"Portable manifests must use relative artifact paths: {value}")
    root = manifest_path.parent.resolve()
    resolved = (root / p).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest artifact escapes its output directory: {value}") from exc
    return resolved


def self_test() -> None:
    """Exercise event graphs, exact W grid and relative-path rejection without real data."""
    polygon = [[0, 0], [10, 0], [10, 30], [0, 30]]
    gt = np.asarray([[10, 7, 11, 4, 2, 4], [11, 7, 8, 4, 2, 4]], dtype=float)
    mot = np.asarray([[10, 9, 11, 4, 2, 4], [11, 9, 8, 4, 2, 4]], dtype=float)
    raw = b"10,9,11,4,2,4,1,-1,-1,-1\n11,9,8,4,2,4,1,-1,-1,-1\n"
    scored = score_case(gt, mot, polygon, 1000)
    independent = verify_mot(raw, gt, polygon, 1000)
    scored["mot_sha256"] = hashlib.sha256(raw).hexdigest()
    assert_score(independent, scored)
    assert scored["window"]["l1_by_offset_in_out"] == [[0, 0]] * 30
    assert scored["iou"]["0.5"]["tp"] == 1 and scored["iou"]["0.3"]["tp"] == 1
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "clip.txt").write_bytes(raw)
        assert relative_artifact(root / "manifest.json", "clip.txt").read_bytes() == raw
        for bad in ("../outside", str(root / "clip.txt")):
            try:
                relative_artifact(root / "manifest.json", bad)
            except ValueError:
                pass
            else:
                raise AssertionError("Portable path validator accepted absolute/traversal input")
    print("Synthetic score, official graph cross-check, 30 offsets, and manifest paths passed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifests", type=Path, nargs="*")
    ap.add_argument("--repo", type=Path, help="BeeTrack checkout; inferred from cwd/script when omitted")
    ap.add_argument("--runs", type=Path, help="runs directory; defaults to <repo>/runs")
    ap.add_argument("--zones", type=Path, help="zones.json; defaults to <repo>/beetrack/zones.json")
    ap.add_argument("--out", type=Path, help="score output directory")
    ap.add_argument("--prereg", help="must match each manifest's prereg_verified value")
    ap.add_argument("--self-test", action="store_true", help="run synthetic scorer/path checks only")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.manifests:
        ap.error("provide at least one portable manifest")
    if not args.out or not args.prereg:
        ap.error("--out and --prereg are required")
    repo = find_repo(args.repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    runs = (args.runs or repo / "runs").resolve()
    zones_path = (args.zones or repo / "beetrack" / "zones.json").resolve()
    zones = json.loads(zones_path.read_bytes())
    records = []
    keys = set()
    manifest_hashes = {}
    complete_flags = []
    for manifest_arg in args.manifests:
        manifest_path = manifest_arg.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cells = manifest.get("cells")
        if manifest.get("schema") != 126 or not isinstance(cells, list) or not cells:
            raise ValueError(f"Expected witness manifest schema 126 with cells: {manifest_path}")
        if manifest.get("prereg_verified") != args.prereg:
            raise ValueError(f"Preregistration hash mismatch: {manifest_path}")
        manifest_hashes[manifest_path.parent.name] = sha(manifest_path)
        complete_flags.append(manifest.get("complete") is True)
        for c in cells:
            allowed = {"gt": set("GTECK"), "real": set("UMETK")}
            if (c.get("seq") not in HEADS or c.get("family") not in allowed or
                    c.get("arm") not in allowed.get(c.get("family"), set()) or not c.get("mot") or
                    str(c.get("fold")) != c["seq"].split("-")[1]):
                raise ValueError(f"Invalid dev witness cell: {c}")
            key = (c["seq"], c["family"], c["arm"], c.get("seed"))
            if key in keys:
                raise ValueError(f"Duplicate witness cell {key}")
            keys.add(key)
            p = relative_artifact(manifest_path, c["mot"])
            if not p.is_file():
                raise FileNotFoundError(p)
            if sha(p) != c.get("mot_sha256"):
                raise ValueError(f"MOT digest mismatch: {c['mot']}")
            archive = relative_artifact(manifest_path, c.get("archive", ""))
            if not archive.is_file() or sha(archive) != c.get("archive_sha256"):
                raise ValueError(f"Archive missing or digest mismatch: {c.get('archive')}")
            records.append(score_one(c, p, args.prereg, runs, zones))
    required = {(head, family, arm, seed) for head in HEADS
                for family, arms, seeds in (("gt", "GTECK", (None,)), ("real", "UMETK", (0, 1, 2)))
                for arm in arms for seed in seeds}
    gate_ready = (set(keys) >= required and all(complete_flags))
    def mean_w(family: str, arm: str, case: str) -> float | None:
        vals = [r["window"]["w30_percent"] for r in records
                if r["family"] == family and r["arm"] == arm and r["case"] == case]
        return float(np.mean(vals)) if vals else None

    historic = {
        "gt_G_w30_percent": {head: mean_w("gt", "G", head) for head in HEADS},
        "real_U_w30_percent_mean_by_head": {head: mean_w("real", "U", head) for head in HEADS},
        "real_M_w30_percent_mean_by_head": {head: mean_w("real", "M", head) for head in HEADS},
        "rounded_to_0_01_matches_prior_G_U_M": {
            "gt_G": {"BEE24-33-head": round(mean_w("gt", "G", "BEE24-33-head") or 0, 2) == 29.17,
                     "BEE24-35-head": round(mean_w("gt", "G", "BEE24-35-head") or 0, 2) == 7.86},
            "real_U": {"BEE24-33-head": round(mean_w("real", "U", "BEE24-33-head") or 0, 2) == 23.27,
                       "BEE24-35-head": round(mean_w("real", "U", "BEE24-35-head") or 0, 2) == 17.76},
            "real_M": {"BEE24-33-head": round(mean_w("real", "M", "BEE24-33-head") or 0, 2) == 39.55,
                       "BEE24-35-head": round(mean_w("real", "M", "BEE24-35-head") or 0, 2) == 16.24}},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for r in records:
        seed_tag = "null" if r["seed"] is None else str(r["seed"])
        dest = args.out / r["case"] / f"{r['family']}_{r['arm']}_seed{seed_tag}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"format": "study126-score-v1", "manifest_sha256": manifest_hashes,
               "prereg": args.prereg, "gate_ready_full_40_cells": gate_ready,
               "paired": summarize(records, gate_ready), "historical_score_checks": historic,
               "oracle_gt": {"w30_by_head": {r["case"]: r["window"]["w30_percent"] for r in records
                                               if r["family"] == "gt" and r["arm"] == "G"},
                             "35_head_below_5_05": next((r["window"]["w30_percent"] < 5.05 for r in records
                                                          if r["case"] == "BEE24-35-head" and
                                                          r["family"] == "gt" and r["arm"] == "G"), None),
                             "all_gt_arm_w30_by_head": {r["case"]: {x["arm"]: x["window"]["w30_percent"]
                                                                    for x in records if x["case"] == r["case"] and
                                                                    x["family"] == "gt"}
                                                       for r in records if r["family"] == "gt"}},
               "records": [dict(case=r["case"], condition=r["condition"], family=r["family"],
                                 arm=r["arm"], seed=r["seed"],
                                 sha256=sha(args.out / r["case"] /
                                            f"{r['family']}_{r['arm']}_seed{'null' if r['seed'] is None else r['seed']}.json"))
                           for r in records]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
