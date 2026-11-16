"""Freeze aligned no-white P0 support on BEE24-10/15 before scoring any truth."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np
import scipy

from beetrack.common import ROOT, RUNS
from beetrack.evaluate import ZONES

sys.path.insert(0, str(ROOT / "tools" / "study108"))
from count_supported import count_supported, read_events
from support import _matches_by_lag, _observations


CLIPS = {10: 250, 15: 240}
SEEDS = (0, 1, 2)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quota_masks(events, retained, clip_id, anchor_seed):
    by_id = {row[0]: row[1] for row in events}
    keep = set(retained)
    if len(by_id) != len(events) or len(keep) != len(retained) or not keep.issubset(by_id):
        raise ValueError("Anchor masks require unique, valid anchor event IDs")
    pools = [np.array(sorted(pid for pid, f in by_id.items() if f * sign > 0), dtype=np.int64)
             for sign in (1, -1)]
    sizes = [len(keep.intersection(pool.tolist())) for pool in pools]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([109, clip_id, anchor_seed])))
    result = []
    for _ in range(99):
        # Call choice in each direction, including full or empty budgets.
        chosen = [rng.choice(pool, size=size, replace=False).tolist() for pool, size in zip(pools, sizes)]
        result.append(sorted(chosen[0] + chosen[1]))
    return result


def anchor_masks(tracks, clip_id, nframes):
    observations = {s: _observations(tracks[str(s)]["predictions"], nframes) for s in SEEDS}
    result = {}
    for seed in SEEDS:
        buddies = [s for s in SEEDS if s != seed]
        pairs = []
        retained = set()
        for buddy in buddies:
            # §108's exact assignment; a short clip supplies aligned times only.
            matched, aligned_pairs = _matches_by_lag(observations[seed], observations[buddy], nframes, ())
            pairs.append(aligned_pairs)
            retained.update(matched[0])
        kept = sorted(retained)
        result[str(seed)] = dict(anchor_seed=seed, buddy_seeds=buddies, pairs=pairs, retained=kept,
                                 quota_masks=quota_masks(tracks[str(seed)]["predictions"], kept, clip_id, seed))
    return result


def freeze(prereg):
    if re.fullmatch("[0-9a-f]{40}", prereg) is None:
        raise ValueError("Provide the full preregistration commit SHA")
    zone_bytes = ZONES.read_bytes()
    zones = json.loads(zone_bytes)
    clips = {}
    for clip_id, nframes in CLIPS.items():
        clip = f"BEE24-{clip_id}"
        poly = zones[clip]
        tracks, paths = {}, {}
        for seed in SEEDS:
            tag = f"loho_noW_s{seed}"
            path = RUNS / "trackers" / "val" / f"q_lk_{tag}" / "data" / (clip + ".txt")
            paths[seed] = path
            checksum = sha(path)
            tracks[str(seed)] = dict(seed=seed, tag=tag, mot_path=path.relative_to(ROOT).as_posix(),
                                     mot_sha256=checksum, predictions=read_events(path, poly, nframes))
        anchors = anchor_masks(tracks, clip_id, nframes)
        for seed in SEEDS:
            anchor = anchors[str(seed)]
            first, second = anchor["buddy_seeds"]
            native = count_supported(paths[seed], paths[first], paths[second], poly, nframes)
            expected = {p: {} for p in anchor["retained"]}
            for role, pairs in zip(("buddy1", "buddy2"), anchor["pairs"]):
                for p, b in pairs:
                    expected[p][role] = b
            if {event["anchor_id"]: event["supporters"] for event in native["events"]} != expected:
                raise ValueError("Native count_supported disagrees with the replayable support pairs")
            anchor.update(anchor_counts=native["anchor_counts"], kept_counts=native["kept_counts"])
        if any(sha(paths[s]) != tracks[str(s)]["mot_sha256"] for s in SEEDS):
            raise ValueError("Input MOT changed during freeze")
        clips[clip] = dict(clip_id=clip_id, nframes=nframes, polygon=poly, tracks=tracks, anchors=anchors)
    return dict(study=109, prereg=prereg, primary_anchor=0, fps=25, clips=clips,
                zones_path=ZONES.relative_to(ROOT).as_posix(), zones_sha256=hashlib.sha256(zone_bytes).hexdigest(),
                descriptor="resolved MOT ID, signed clip-local frame (+in/-out), x1,y1,x2,y2 in pixels",
                versions={"numpy": np.__version__, "scipy": scipy.__version__})


def write_once(output, record):
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode()
    with TemporaryDirectory(dir=output.parent) as tmp:
        part = Path(tmp) / "white_masks.json"
        part.write_bytes(data)
        try:
            os.link(part, output)
        except FileExistsError:
            if output.read_bytes() != data:
                raise ValueError(f"{output} already contains a different freeze")
    return hashlib.sha256(data).hexdigest()


def replay(record):
    assert (record["study"], record["fps"], record["primary_anchor"]) == (109, 25, 0)
    assert set(record["clips"]) == {f"BEE24-{h}" for h in CLIPS}
    for clip, row in record["clips"].items():
        num = row["clip_id"]
        assert clip == f"BEE24-{num}" and row["nframes"] == CLIPS[num]
        masks = anchor_masks(row["tracks"], num, CLIPS[num])
        for seed, result in masks.items():
            events = row["tracks"][seed]["predictions"]
            kept = set(result["retained"])
            for name, selected in (("anchor_counts", events), ("kept_counts", [e for e in events if e[0] in kept])):
                result[name] = {"in": sum(e[1] > 0 for e in selected), "out": sum(e[1] < 0 for e in selected)}
        assert masks == row["anchors"]


def synthetic():
    # Include tied candidate boxes, simultaneous opposing directions and an unsupported event.
    data = [
        [[10,30,0,0,2,1], [11,30,0,0,2,1], [12,-30,0,0,2,1], [13,35,4,0,6,1]],
        [[50,35,0,0,1,1], [51,-30,0,0,1,1], [52,41,4,0,6,1]],
        [[100,35,4,0,6,1], [101,-35,0,0,.99,1]],
    ]
    tracks = {str(s): {"predictions": rows} for s, rows in enumerate(data)}
    masks = anchor_masks(tracks, 10, CLIPS[10])
    main = masks["0"]
    assert main["retained"] == [10,12,13]
    assert main["pairs"] == [[[10,50],[12,51]],[[13,100]]]
    assert len(main["quota_masks"]) == 99
    assert all(len(row) == 3 and 12 in row and len(set(row)) == 3 for row in main["quota_masks"])
    assert masks == anchor_masks(tracks, 10, CLIPS[10])
    assert main["quota_masks"] != anchor_masks(tracks, 15, CLIPS[15])["0"]["quota_masks"]
    for clip_id, nf in CLIPS.items():
        for seed in SEEDS:
            assert quota_masks([], [], clip_id, seed) == [[]] * 99
            assert quota_masks(data[0], [10,11,12,13], clip_id, seed) == [[10,11,12,13]] * 99
        assert anchor_masks({str(s): {"predictions": []} for s in SEEDS}, clip_id, nf)["0"]["retained"] == []
    print("Aligned matches, tie order, unsynchronized directions, ROI threshold, and 99 directional quota masks passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    cmd = sub.add_parser("freeze")
    cmd.add_argument("output", type=Path)
    cmd.add_argument("--prereg", required=True)
    sub.add_parser("synthetic")
    cmd = sub.add_parser("replay")
    cmd.add_argument("input", type=Path)
    args = p.parse_args()
    if args.mode == "synthetic":
        synthetic()
    elif args.mode == "freeze":
        record = freeze(args.prereg)
        replay(record)
        print(f"{args.output} SHA256 {write_once(args.output, record)}")
    else:
        replay(json.loads(args.input.read_bytes()))
        print(f"Replayed both clips and three anchor rotations; SHA256 {sha(args.input)}")
