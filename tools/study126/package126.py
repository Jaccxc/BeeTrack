"""Package Study 126 score JSON records into path-free replay evidence."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

OFFSETS = list(range(0, 750, 25))


def graph_offsets():
    return OFFSETS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scores", type=Path, help="directory tree of scorer JSON records")
    ap.add_argument("output", type=Path, help="gzip JSON output path")
    args = ap.parse_args()
    records = []
    for path in sorted(args.scores.glob("*/*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if "graph_evidence" not in r:
            continue
        ev = r["graph_evidence"]
        graph = {k: ev[k] for k in ("format", "nframes", "gt", "edges05", "edges03", "inputs_sha256") if k in ev}
        if "edges05" not in graph and "edges" in ev:
            graph["edges05"], graph["edges03"] = ev["edges"]["0.5"], ev["edges"]["0.3"]
        graph["predictions"] = [row[:2] for row in ev["predictions"]]
        window = r["window"]
        record = {"case": r["case"], "family": r["family"], "arm": r["arm"],
                  "nframes": r["nframes"], "gt_events": r["gt_events"],
                  "window": {"width_frames": window["width_frames"],
                             "offsets_frames": window.get("offsets_frames", graph_offsets()),
                             "gt_denominator": window["gt_denominator"],
                             "integer_absolute_error_total": window["integer_absolute_error_total"]},
                  "official_iou": {t: {k: r["official_iou"][t][k] for k in ("tp", "fp", "fn")}
                                   for t in ("0.5", "0.3")}, "mot_sha256": r["provenance"]["mot_sha256"],
                  "graph_evidence": graph}
        if r.get("seed") is not None:
            record["seed"] = r["seed"]
        records.append(record)
    if not records:
        raise SystemExit("no graph evidence records found")
    bundle = {"format": "study126-replay-bundle-v1", "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as z:
            z.write((json.dumps(bundle, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
