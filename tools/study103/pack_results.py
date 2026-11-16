"""Package calculator inputs from the preregistered §103 runs, with no local paths."""

import argparse
from fractions import Fraction
import json
from pathlib import Path


def events(pairs):
    return sorted((frame * direction for frame, direction in pairs), key=lambda v: (abs(v), v))


def short(obj):
    if isinstance(obj, float):
        return round(obj, 5)
    if isinstance(obj, dict):
        return {key: short(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [short(value) for value in obj]
    return obj


def pack(source):
    clips = {}
    cases = []
    for hive in (33, 35):
        for seed in range(9):
            p = json.loads((source / "p0" / f"{hive}_{seed}.json").read_text(encoding="utf-8"))
            assert (p["hive"], p["seed"]) == (hive, seed)
            clip = {"frames": p["nframes"], "gt": events(p["gt_events"])}
            assert clips.setdefault(str(hive), clip) == clip
            candidates = [dict(pair=c["pair"], before=events(c["before"]), after=events(c["after"]), kind=c["kind"])
                          for c in p["candidates"]]
            cases.append(dict(hive=hive, seed=seed, events=events(p["p0_events"]), method="P0",
                              phase=p["phase"], census=p["census"], candidates=candidates, arms=p["arms"]))
    controls = json.loads((source / "comparisons.json").read_text(encoding="utf-8"))
    raw = {(r["hive"], r["seed"]): events(r["pred_events"]) for r in controls if r["method"] == "ByteTrack"}
    for r in controls:
        assert clips[str(r["hive"])] == {"frames": r["nframes"], "gt": events(r["gt_events"])}
        case = dict(hive=r["hive"], seed=r["seed"], method=r["method"], phase=r["phase_summary"],
                    reference_error=r["published_error"])
        stream = events(r["pred_events"])
        if r["method"] == "calibrated ByteTrack":
            assert stream == raw[r["hive"], r["seed"]]
            fractions = [Fraction(value).limit_denominator(10000) for value in r["scale"]]
            assert all(abs(float(q) - value) < 1e-14 for q, value in zip(fractions, r["scale"]))
            case["scale_ratio"] = [[q.numerator, q.denominator] for q in fractions]
        else:
            assert r["scale"] == [1.0, 1.0]
            case["events"] = stream
        cases.append(case)
    return dict(study=103, preregistration="e62e337760ea91526a23f5590804b9b991676734",
                units="Signed frame number: positive=in, negative=out. Frames start at 1; 25 fps.",
                calibration="Calibrated ByteTrack reuses its seed's ByteTrack events. Scale ratios [numerator,denominator] are ordered in,out.",
                caveat="Cached MOT boxes and the dev GT are required to rediscover local identity exchanges or match events by IoU. This archive replays the count and phase controls from events, and F1 arithmetic from matched counts.",
                clips=clips, cases=[short(case) for case in cases])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = pack(args.source)
    cases = data.pop("cases")
    encode = lambda item: json.dumps(item, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    content = encode(data)[:-1] + ',\n"cases":[\n' + ",\n".join(map(encode, cases)) + "\n]}\n"
    args.output.write_text(content, encoding="utf-8", newline="\n")
