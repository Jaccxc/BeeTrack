"""Count anchor events supported by either of two other same-clip P0 track files."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from beetrack.evaluate import load_tracks
from inputs108 import describe
from support import _matches_by_lag, _observations


def read_polygon(path, clip):
    points = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(points, dict):
        if clip not in points:
            raise ValueError("Supply --clip with a key from the polygon map")
        points = points[clip]
    polygon = np.asarray(points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("The entrance polygon requires at least three finite [x,y] vertices")
    x, y = polygon.T
    if np.dot(x, np.roll(y,-1)) == np.dot(y, np.roll(x,-1)):
        raise ValueError("The entrance polygon has zero area")
    return polygon


def read_events(path, polygon, nframes):
    tracks = load_tracks(path) if path.stat().st_size else np.empty((0, 6))
    if tracks.shape[1] != 6 or not np.isfinite(tracks).all():
        raise ValueError(f"Expected finite frame,id,x,y,width,height MOT columns in {path}")
    if np.any(tracks[:, :2] != np.round(tracks[:, :2])) or np.any(tracks[:, 4:6] <= 0):
        raise ValueError(f"Frame and ID must be integral; boxes require positive area in {path}")
    if np.any((tracks[:, 0] < 1) | (tracks[:, 0] > nframes)):
        raise ValueError(f"MOT frames must be in 1..{nframes} in {path}")
    return describe(tracks, polygon)


def count_supported(anchor_mot, buddy1_mot, buddy2_mot, polygon, nframes):
    if nframes < 1:
        raise ValueError("Supply a positive clip length")
    anchor, buddy1, buddy2 = (read_events(path, polygon, nframes)
                              for path in (anchor_mot, buddy1_mot, buddy2_mot))
    a = _observations(anchor, nframes)
    support = {}
    for role, events in (("buddy1", buddy1), ("buddy2", buddy2)):
        _, pairs = _matches_by_lag(a, _observations(events, nframes), nframes, ())
        for anchor_id, buddy_id in pairs:
            support.setdefault(anchor_id, {})[role] = buddy_id
    anchor_counts = {"in": 0, "out": 0}
    kept_counts = {"in": 0, "out": 0}
    result = []
    for track_id, signed, *box in anchor:
        direction = "in" if signed > 0 else "out"
        anchor_counts[direction] += 1
        if track_id in support:
            kept_counts[direction] += 1
            result.append(dict(anchor_id=track_id, frame=abs(signed), direction=direction,
                               box_xyxy=box, supporters=support[track_id]))
    result.sort(key=lambda e: (e["frame"], e["anchor_id"]))
    return dict(nframes=nframes, anchor_counts=anchor_counts, kept_counts=kept_counts, events=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, required=True, help="number of 1-based clip frames")
    parser.add_argument("--polygon", type=Path, required=True, help="JSON polygon or a map of clip names to polygons")
    parser.add_argument("--clip", help="entry in a polygon map")
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--buddy1", type=Path, required=True)
    parser.add_argument("--buddy2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.polygon,args.anchor,args.buddy1,args.buddy2)}:
        parser.error("The output must have a different path from every input")
    result = count_supported(args.anchor, args.buddy1, args.buddy2,
                             read_polygon(args.polygon, args.clip), args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        part = Path(tmp) / "events.json"
        part.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        part.replace(args.output)
    print(f"Anchor {result['anchor_counts']}; supported {result['kept_counts']}")


if __name__ == "__main__":
    main()
