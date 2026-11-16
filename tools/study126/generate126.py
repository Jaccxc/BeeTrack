"""Portable CPU generator for the preregistered §126 masks and fixed-P0 MOTs.

Run from a BeeTrack checkout. Only dev heads 33/35 and detector seeds 0–2 are opened.
The explicit strict-linker JSON must be the preregistered m_dev_strict.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path.cwd().resolve()
if (REPO / "beetrack").is_dir():
    sys.path.insert(0, str(REPO))

FOLDS = {"35": "BEE24-35-head", "33": "BEE24-33-head"}
PREREG = "c9ecd32a5632cd9420859c2718ffafad8dd38797"
STRICT_SHA = "fdd788e0dba0e6fa1c2d06feec3127bf84d5e7b2d214cc0f793c452270403dcf"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def keyed_control(pool, temporal, clip):
    selected = np.zeros(len(pool), bool)
    for frame in np.unique(pool[:, 0]):
        candidates = np.flatnonzero(pool[:, 0] == frame)
        budget = int(temporal[candidates].sum())
        ordered = sorted(candidates, key=lambda r: (
            hashlib.sha256(f"126|{clip}|{int(frame)}|{r}".encode("utf-8")).digest(), int(r)))
        selected[ordered[:budget]] = True
    return selected


def official_provenance(raw, prepared):
    from beetrack.evaluate import by_track, merge_handovers
    tagged = np.c_[raw[:, :6], np.arange(len(raw))]
    _, first = np.unique(tagged[:, 0] * 1e6 + tagged[:, 1], return_index=True)
    duplicates = np.setdiff1d(np.arange(len(raw)), first)
    tagged[duplicates, 1] = tagged[:, 1].max() + 1 + np.arange(len(duplicates))
    tagged = merge_handovers(tagged)
    rounded = tagged[:, :6].copy()
    rounded[:, 2:6] = np.array([[float(f"{x:.2f}") for x in r] for r in rounded[:, 2:6]])
    if not np.array_equal(rounded, prepared[:, :6]):
        raise ValueError("Prepared GT provenance does not exactly reconstruct; stop")
    tokens = tagged[:, 6].astype(int)
    if not np.array_equal(np.sort(tokens), np.arange(len(raw))):
        raise ValueError("Source row tokens are not one-to-one")
    official = by_track(np.c_[prepared[:, :6], tokens])
    segment = np.empty(len(raw), dtype=np.int64)
    segment[official[:, 6].astype(int)] = official[:, 1].astype(np.int64)
    counts = np.unique(official[:, :2], axis=0, return_counts=True)[1]
    if counts.max(initial=0) > 1:
        raise ValueError("An official segment has simultaneous boxes; provenance is ambiguous")
    return official, segment, tokens, duplicates


def support_masks(raw, prepared, polygon):
    from beetrack.evaluate import zone_tracks
    official, segment, tokens, duplicates = official_provenance(raw, prepared)
    event = np.zeros(len(raw), bool)
    temporal = np.zeros(len(raw), bool)
    events = []
    for tid, (direction, _start, _end, crossings) in zone_tracks(official, polygon).items():
        if not direction:
            continue
        arrival = [j for j, d in crossings if d == direction][-1]
        frame = int(official[arrival, 0])
        selected = segment == tid
        event |= selected
        temporal |= selected & (np.abs(raw[:, 0] - frame) <= 12)
        events.append(dict(segment=int(tid), direction=direction, arrival=frame))
    if np.any(temporal & ~event):
        raise AssertionError("Temporal support must be a subset of event segments")
    return {"G": np.ones(len(raw), bool), "E": event, "T": temporal, "C": ~event}, dict(
        official=official, segment=segment, prepared_to_raw=tokens, duplicate_rows=duplicates, events=events)


def full_match(dets, prepared):
    from scipy.optimize import linear_sum_assignment
    from beetrack.track import iou_matrix
    gtboxes = np.c_[prepared[:, 2:4], prepared[:, 2:4] + prepared[:, 4:6]]
    assigned = np.full(len(dets), -1, dtype=np.int64)
    alternatives = np.zeros(len(dets), dtype=np.int64)
    for frame in np.unique(np.r_[dets[:, 0], prepared[:, 0]]):
        d = np.flatnonzero(dets[:, 0] == frame)
        g = np.flatnonzero(prepared[:, 0] == frame)
        if len(d) and len(g):
            matrix = iou_matrix(gtboxes[g], dets[d, 1:5])
            alternatives[d] = (matrix >= .5).sum(axis=0)
            i, j = linear_sum_assignment(-matrix)
            keep = matrix[i, j] >= .5
            assigned[d[j[keep]]] = g[i[keep]]
    return assigned, alternatives


def keyed_control_matched(dets, hit, temporal, clip):
    selected = np.zeros(len(dets), bool)
    for frame in np.unique(dets[:, 0]):
        candidates = np.flatnonzero(hit & (dets[:, 0] == frame))
        budget = int(temporal[dets[:, 0] == frame].sum())
        ordered = sorted(candidates, key=lambda r: (
            hashlib.sha256(f"126|{clip}|{int(frame)}|{r}".encode("utf-8")).digest(), int(r)))
        selected[ordered[:budget]] = True
    return selected


def track_rows(detections, probability):
    """Public repo equivalent of the historical first pass + fixed linker pipeline."""
    from beetrack.link import link
    from beetrack.track import Tracker, drop_short, run_sequence
    stage, _ = run_sequence(detections, 3950, Tracker("link", link=0, min_len=0))
    stage = np.asarray(stage, float).reshape(-1, 7)
    rows = sorted(link(stage, .15, probability).tolist())
    rows = np.asarray(drop_short(rows, 3, 0.), dtype=float).reshape(-1, 7)
    return stage, rows


def make_input(root, fold, seed, prepared, masks, provenance, polygon):
    from beetrack.common import gt_slice
    clip = FOLDS[fold]
    raw = gt_slice(f"BEE24-{fold}", 1, 3950)
    oracle_path = root / "runs" / "dets" / "oracle" / "dev" / f"{clip}.npy"
    oracle = np.load(oracle_path, allow_pickle=False)
    rebuilt = np.c_[raw[:, 0], raw[:, 2:4], raw[:, 2:4] + raw[:, 4:6], np.ones(len(raw))]
    if not np.array_equal(oracle, rebuilt):
        raise ValueError("§122 oracle does not exactly equal the raw source rows")
    if seed is None:
        selected = dict(masks)
        selected["K"] = keyed_control(oracle, masks["T"], clip)
        return oracle, selected, {"oracle": oracle_path.relative_to(root).as_posix(), "oracle_sha256": sha(oracle_path)}
    detector_path = root / "runs" / "dets" / f"loho_no{fold}_s{seed}" / "dev" / f"{clip}.npy"
    dets = np.load(detector_path, allow_pickle=False)
    assigned, alternatives = full_match(dets, prepared)
    hit = assigned >= 0
    selected = {"U": np.ones(len(dets), bool), "M": hit}
    for arm in ("E", "T"):
        keep = np.zeros(len(dets), bool)
        keep[hit] = masks[arm][provenance["prepared_to_raw"][assigned[hit]]]
        selected[arm] = keep
    selected["K"] = keyed_control_matched(dets, hit, selected["T"], clip)
    return dets, selected, {"detector": detector_path.relative_to(root).as_posix(), "detector_sha256": sha(detector_path),
                             "matched_prepared_row0": assigned, "eligible_gt_count": alternatives}


def emit(out, fold, family, arm, seed, pool, selected, probability, extra=None, track=True):
    from beetrack.common import write_mot
    name = f"{fold}_{family}_{arm}" + ("" if seed is None else f"_s{seed}")
    chosen = pool[selected]
    stage, rows = track_rows(chosen, probability) if track else (None, None)
    mot, archive = out / f"{name}.txt", out / f"{name}.npz"
    if track:
        write_mot(mot, rows)
        np.savez_compressed(archive, input=chosen, selected_row0=np.flatnonzero(selected), stage1=stage,
                            output=rows, **(extra or {}))
    else:
        np.savez_compressed(archive, input=chosen, selected_row0=np.flatnonzero(selected), **(extra or {}))
    return name, mot, archive, rows


def run(args):
    from beetrack.common import RUNS
    from beetrack.evaluate import ZONES
    from beetrack.link import model
    root = Path.cwd().resolve()
    weights = args.weights.resolve()
    if sha(weights) != STRICT_SHA:
        raise ValueError("--weights must be the pinned m_dev_strict.json")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    probability = model(json.loads(weights.read_bytes()))
    zones = json.loads(ZONES.read_text(encoding="utf-8"))
    manifest = dict(schema=126, prereg_verified=PREREG, cells=[], sources={
        "strict_model": {"sha256": sha(weights)},
        "binding": "beetrack.track.run_sequence(D,3950,Tracker('link',link=0,min_len=0)); "
                   "beetrack.link.link(stage,0.15,strict_prob); drop_short(...,3,0.0)"})
    for fold in args.folds:
        clip = FOLDS[fold]
        raw = __import__("beetrack.common", fromlist=["gt_slice"]).gt_slice(f"BEE24-{fold}", 1, 3950)
        gtpath = RUNS / "gt" / "dev" / clip / "gt" / "gt.txt"
        prepared = np.loadtxt(gtpath, delimiter=",", ndmin=2)
        masks, provenance = support_masks(raw, prepared, zones[clip])
        oracle_path = RUNS / "dets" / "oracle" / "dev" / f"{clip}.npy"
        oracle = np.load(oracle_path, allow_pickle=False)
        masks["K"] = keyed_control(oracle, masks["T"], clip)
        np.savez_compressed(out / f"{fold}_provenance.npz", raw=raw, prepared=prepared,
                            **{k: v for k, v in provenance.items() if k != "events"},
                            **{f"mask_{k}": v for k, v in masks.items()})
        json_save(out / f"{fold}_events.json", provenance["events"])
        manifest["sources"][clip] = dict(oracle=oracle_path.relative_to(root).as_posix(),
                                         oracle_sha256=sha(oracle_path),
                                         prepared=gtpath.relative_to(root).as_posix(),
                                         prepared_sha256=sha(gtpath))
        for arm in ("G", "T", "E", "C", "K"):
            name, mot, archive, _ = emit(out, fold, "gt", arm, None, oracle, masks[arm], probability)
            manifest["cells"].append(cell_record(out, fold, clip, "gt", arm, None, mot, archive))
            json_save(out / "manifest.json", manifest)
            print(name, "sealed", flush=True)
        for seed in args.seeds:
            dets, real_masks, lineage = make_input(root, fold, seed, prepared, masks, provenance, zones[clip])
            source_key = f"{clip}_s{seed}"
            manifest["sources"][source_key] = {k: v for k, v in lineage.items() if not isinstance(v, np.ndarray)}
            for arm in ("U", "M", "E", "T", "K"):
                extra = {k: v for k, v in lineage.items() if isinstance(v, np.ndarray)}
                name, mot, archive, _ = emit(out, fold, "real", arm, seed, dets, real_masks[arm], probability, extra)
                manifest["cells"].append(cell_record(out, fold, clip, "real", arm, seed, mot, archive))
                json_save(out / "manifest.json", manifest)
                print(name, "sealed", flush=True)
    manifest["complete"] = True
    json_save(out / "manifest.json", manifest)


def cell_record(root, fold, clip, family, arm, seed, mot, archive):
    return dict(fold=fold, seed=seed, seq=clip, family=family, arm=arm,
                mot=mot.relative_to(root).as_posix(), archive=archive.relative_to(root).as_posix(),
                mot_sha256=sha(mot), archive_sha256=sha(archive))


def verify_existing(args):
    """Check every extant source-row selection and recompute only representative T/U MOTs."""
    from beetrack.common import RUNS, write_mot
    from beetrack.evaluate import ZONES
    from beetrack.link import model
    root, old, out = Path.cwd().resolve(), args.verify_existing.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError("Choose a new output directory")
    out.mkdir(parents=True)
    weights = args.weights.resolve()
    if sha(weights) != STRICT_SHA:
        raise ValueError("--weights must be the pinned m_dev_strict.json")
    probability = model(json.loads(weights.read_bytes()))
    zones = json.loads(ZONES.read_text(encoding="utf-8"))
    index = json.loads((old / "manifest.json").read_bytes())
    cells = index["cells"]
    checked = []
    for fold in args.folds:
        clip = FOLDS[fold]
        from beetrack.common import gt_slice
        raw = gt_slice(f"BEE24-{fold}", 1, 3950)
        gtpath = RUNS / "gt" / "dev" / clip / "gt" / "gt.txt"
        prepared = np.loadtxt(gtpath, delimiter=",", ndmin=2)
        masks, provenance = support_masks(raw, prepared, zones[clip])
        oracle_path = RUNS / "dets" / "oracle" / "dev" / f"{clip}.npy"
        oracle = np.load(oracle_path, allow_pickle=False)
        masks["K"] = keyed_control(oracle, masks["T"], clip)
        real_cache = {}
        for cell in [c for c in cells if c["fold"] == fold]:
            family, arm, seed = cell["family"], cell["arm"], cell["seed"]
            if family == "gt":
                pool, selected = oracle, masks[arm]
            else:
                if seed not in real_cache:
                    real_cache[seed] = make_input(root, fold, seed, prepared, masks, provenance, zones[clip])
                pool, all_selected, _lineage = real_cache[seed]
                selected = all_selected[arm]
            archive = np.load(old / Path(cell["archive"]).name, allow_pickle=False)
            expected_ids = np.flatnonzero(selected)
            expected_rows = pool[selected]
            if not np.array_equal(archive["selected_row0"], expected_ids):
                raise ValueError(f"Selected-row mismatch: {cell['family']}/{cell['arm']}/{seed}")
            if not np.array_equal(archive["input"], expected_rows):
                raise ValueError(f"Selected-input mismatch: {cell['family']}/{cell['arm']}/{seed}")
            checked.append(f"{fold}/{family}/{arm}/{seed}")
        # Recompute GT-T for each fold. Recompute real-U where the old lane contains it.
        for family, arm, seed in (("gt", "T", None), ("real", "U", 0)):
            cell = next((c for c in cells if c["fold"] == fold and c["family"] == family
                         and c["arm"] == arm and c["seed"] == seed), None)
            if cell is None:
                continue
            if family == "gt":
                pool, selected = oracle, masks["T"]
            else:
                if seed not in real_cache:
                    real_cache[seed] = make_input(root, fold, seed, prepared, masks, provenance, zones[clip])
                pool, real, _ = real_cache[seed]
                selected = real["U"]
            stage, rows = track_rows(pool[selected], probability)
            reference = old / Path(cell["mot"]).name
            temp = out / f"{fold}_{family}_{arm}_s{seed if seed is not None else 'na'}.txt"
            write_mot(temp, rows)
            if temp.read_bytes() != reference.read_bytes():
                raise ValueError(f"Public tracker MOT differs from saved output: {reference.name}")
            original = np.load(old / Path(cell["archive"]).name, allow_pickle=False)
            if not np.array_equal(stage, original["stage1"]) or not np.array_equal(rows, original["output"]):
                raise ValueError(f"Public tracker stage/output differs: {reference.name}")
            print(f"MOT exact: {reference.name}")
    receipt = dict(schema=126, old_lane=old.name, selected_index_checks=checked,
                   public_tracker_exact_mot_count=len(list(out.glob("*.txt"))), strict_sha256=sha(weights),
                   note="only extant cells; no missing cells were regenerated")
    json_save(out / "portable_equivalence.json", receipt)
    print(f"Verified {len(checked)} available selection cells; representative public P0 MOTs exact.")


def self_test():
    raw = np.array([[1,1,20,4,2,2],[2,1,5,4,2,2],[3,2,5,4,2,2],
                    [4,2,4,4,2,2],[1,3,50,4,2,2],[2,3,50,4,2,2]], float)
    from beetrack.evaluate import merge_handovers
    prepared = merge_handovers(raw)
    _, segment, _, duplicates = official_provenance(raw, prepared)
    assert segment[0] == segment[3] and segment[0] != segment[4] and len(duplicates) == 0
    masks, _ = support_masks(raw, prepared, [[0,0],[10,0],[10,10],[0,10]])
    assert masks["E"].tolist() == [True,True,True,True,False,False]
    pool = np.c_[raw[:,0], raw[:,2:4], raw[:,2:4]+raw[:,4:6], np.ones(len(raw))]
    control = keyed_control(pool, masks["T"], "synthetic")
    for frame in np.unique(pool[:,0]):
        assert control[pool[:,0] == frame].sum() == masks["T"][pool[:,0] == frame].sum()
    matched, _ = full_match(pool, prepared)
    assert np.all(matched >= 0)
    assert np.array_equal(keyed_control_matched(pool, np.ones(len(pool), bool), masks["T"], "synthetic"), control)
    synthetic = np.asarray([[frame, x, 5, x+10, 15, .8] for frame in (2,3,4,6,7,8)
                            for x in (8,9,60)], dtype=np.float32)
    probability = lambda features: np.full(len(features), .9)
    stage, result = track_rows(synthetic, probability)
    from tools.study104.track_reverse import track_arms
    reference = dict(track_arms(synthetic, 3950, probability))["F"]
    assert np.array_equal(result, reference) and stage.ndim == 2 and stage.shape[1] == 7
    print("Synthetic provenance, masks, Hungarian match, and public fixed-P0 pipeline passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--weights", type=Path, help="explicit preregistered m_dev_strict.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(3), default=[0,1,2])
    parser.add_argument("--verify-existing", type=Path, help="verify against an existing saved lane; no full regeneration")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.weights or not args.output:
            parser.error("--weights and --output are required")
        if args.verify_existing:
            verify_existing(args)
        else:
            run(args)


if __name__ == "__main__":
    main()
