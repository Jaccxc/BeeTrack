"""Load comparison tracks and reconstruct §95's full-precision source calibration."""

import configparser
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from beetrack.common import ROOT, RUNS
from beetrack.evaluate import ZONES, by_track, load_tracks, zone_crossings


@lru_cache(maxsize=1)
def _zones():
    return json.loads(ZONES.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _source_gt_totals(split, sequence):
    path = RUNS / "gt" / split / sequence / "gt" / "gt.txt"
    return zone_crossings(load_tracks(path), _zones()[sequence])


def _source_bt_rows(tag, split, sequence, cache):
    path = RUNS / "trackers" / split / f"q_bt_{tag}" / "data" / f"{sequence}.txt"
    if path.is_file():
        return load_tracks(path)
    if cache is not None:
        cached = Path(cache) / f"q_bt_{tag}_{split}_{sequence}.txt"
        if cached.is_file():
            return load_tracks(cached)

    # §95 generated a subset of val tracks on demand from archived detections.
    from beetrack.track import Tracker, run_sequence

    cfg = configparser.ConfigParser()
    cfg.read(RUNS / "gt" / split / sequence / "seqinfo.ini", encoding="utf-8")
    nframes = cfg.getint("Sequence", "seqLength")
    dets = np.load(RUNS / "dets" / tag / split / f"{sequence}.npy", allow_pickle=False)
    tracks, _ = run_sequence(dets, nframes, Tracker("bytetrack"))
    if not len(tracks):
        return np.empty((0, 6))
    rows = np.array(tracks, dtype=float)[:, :6].copy()
    rows[:, 2:6] = np.char.mod("%.2f", rows[:, 2:6]).astype(float)
    return rows


def load_comparators(hive, seed, *, calibration_cache=None):
    """Yield (method, target MOT rows, directional count scale) for registered dev heads.

    Optional calibration_cache is §95's t95/cache, used only for missing source
    ByteTrack files. Without it, those files are reconstructed in memory.
    """
    if str(hive) not in ("33", "35") or str(seed) not in tuple(map(str, range(9))):
        raise ValueError("Only registered held-out hives 33/35 and seeds 0..8 are supported")
    hive, seed = int(hive), int(seed)
    other = 35 if hive == 33 else 33
    tag = f"loho_no{hive}_s{seed}"
    target = f"BEE24-{hive}-head"
    trackers = RUNS / "trackers" / "dev"
    raw = by_track(load_tracks(trackers / f"q_bt_{tag}" / "data" / f"{target}.txt"))
    yield "ByteTrack", raw, (1.0, 1.0)

    sources = [("dev", f"BEE24-{other}-head"), ("val", f"BEE24-{other}-tail")]
    sources.extend(("val", f"BEE24-{number}") for number in (10, 15, 24, 28))
    truth = np.zeros(2, dtype=int)
    pred = np.zeros(2, dtype=int)
    for split, sequence in sources:
        truth += _source_gt_totals(split, sequence)
        rows = _source_bt_rows(tag, split, sequence, calibration_cache)
        pred += zone_crossings(rows, _zones()[sequence])
    yield "calibrated ByteTrack", raw, tuple(truth / np.maximum(1, pred))

    if seed <= 2:
        # §99, picks on *seen-hive val* per detector (not best on this dev head).
        threshold = 30 if (hive, seed) == (35, 0) else 20
        selected = f"p99_f2_t{threshold}_h1_{tag}_post"
        rows = by_track(load_tracks(trackers / selected / "data" / f"{target}.txt"))
        yield "TOPICTrack selected", rows, (1.0, 1.0)


def _reference_errors(hive):
    log = (ROOT / "docs" / "experiment_log.md").read_text(encoding="utf-8")
    section = log.split("\n## 95. ", 1)[1].split("\n## 96. ", 1)[0]
    table = section.split(f"留 {hive}(評 {hive}-head;", 1)[1].split("| seed |", 1)[1].split("| 平均(全距)", 1)[0]
    errors = {}
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0].isdigit():
            seed = int(cells[0])
            errors["ByteTrack", seed] = float(cells[4].split("(")[0])
            errors["calibrated ByteTrack", seed] = float(cells[5])
    section = log.split("\n## 99. ", 1)[1].split("\n## 100. ", 1)[0]
    table = section.split(f"| 留 {hive} | 挑到的設定 |", 1)[1]
    line = next(line for line in table.splitlines() if line.startswith("| 公平的重選 |"))
    top = line.strip("|").split("|")[2].strip().split(" / ")
    errors.update({("TOPICTrack selected", seed): float(value) for seed, value in enumerate(top)})
    assert len(errors) == 21
    return errors


def main():
    import argparse

    from phase_control import phase_summary
    from swap_score import event_times

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-cache", type=Path)
    args = parser.parse_args()
    records = []
    for hive in (33, 35):
        target = f"BEE24-{hive}-head"
        gt_dir = RUNS / "gt" / "dev" / target
        cfg = configparser.ConfigParser()
        cfg.read(gt_dir / "seqinfo.ini", encoding="utf-8")
        assert cfg.getint("Sequence", "frameRate") == 25
        nframes = cfg.getint("Sequence", "seqLength")
        polygon = _zones()[target]
        gt = event_times(load_tracks(gt_dir / "gt" / "gt.txt"), polygon)
        expected = _reference_errors(hive)
        for seed in range(9):
            for method, tracks, scale in load_comparators(hive, seed, calibration_cache=args.calibration_cache):
                pred = event_times(tracks, polygon)
                phase = phase_summary(gt, pred, nframes, scale)
                reference = expected[method, seed]
                if round(phase["original"], 2) != reference:
                    raise ValueError(f"{hive} seed {seed} {method}: measured {phase['original']}, reference {reference}")
                records.append(dict(hive=hive, seed=seed, method=method, scale=scale, nframes=nframes,
                                    gt_events=gt.tolist(), pred_events=pred.tolist(), phase_summary=phase,
                                    published_error=reference))
            print(f"verified {hive} seed {seed}", flush=True)
    assert len(records) == 42
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(records, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(f"wrote {len(records)} checked phase comparisons to {args.output}")


if __name__ == "__main__":
    main()
