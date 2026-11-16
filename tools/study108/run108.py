"""Freeze cross-detector event support and quota controls before opening label files."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import scipy

import score108
from support import support_runs


SOURCE = {33: 35, 35: 33}
HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def serialized(record):
    return (json.dumps(record, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data), sha(data)


def write_once(path, record):
    data = serialized(record)
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        candidate = Path(tmp) / path.name
        candidate.write_bytes(data)
        try:
            os.link(candidate, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Existing {path} has different contents")
    return sha(data)


def method_hashes():
    return {name: sha((HERE / name).read_text(encoding="utf-8").encode())
            for name in ("inputs108.py", "support.py", "score108.py", "run108.py")}


def key(fold, seed, part):
    return f"{fold}_{seed}_{part}"


def cases(role):
    parts = ("head", "tail") if role == "source" else ("head",)
    return [(fold, seed, part) for fold in SOURCE for part in parts for seed in range(9)]


def check_fields(row, fold, seed, part, role):
    hive = SOURCE[fold] if role == "source" else fold
    expected = (fold, seed, f"BEE24-{hive}-{part}", 3950 if part == "head" else 1000)
    actual = tuple(row[k] for k in ("fold", "seed", "clip", "nframes"))
    if actual != expected or row["tag"] != f"loho_no{fold}_s{seed}":
        raise ValueError(f"Input {actual} differs from registered case {expected}")


def group_seeds(seed):
    first = 3 * (seed // 3)
    return range(first, first+3)


def controls(inputs):
    anchor, first, second, n = inputs
    support = support_runs(anchor, first, second, n)
    support["quota_masks"] = score108.quota_masks(anchor, support["retained"])
    return support


def inputs_for(predictions, fold, seed, part):
    buddies = [s for s in group_seeds(seed) if s != seed]
    return [predictions[key(fold, s, part)]["predictions"] for s in (seed, *buddies)]


def freeze_masks(args):
    raw, hashes, batches = {}, {}, []
    for fold, seed, part in cases(args.role):
        name = key(fold, seed, part)
        row, checksum = read(args.folder / args.role / "predictions" / f"{name}.json")
        check_fields(row, fold, seed, part, args.role)
        raw[name], hashes[name] = row, checksum
    for fold, seed, part in cases(args.role):
        trio = inputs_for(raw, fold, seed, part)
        batches.append((*trio, raw[key(fold, seed, part)]["nframes"]))
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            result = list(pool.map(controls, batches))
    else:
        result = list(map(controls, batches))
    names = [key(*spec) for spec in cases(args.role)]
    data = dict(role=args.role, prereg=args.prereg, source_sha256=args.source_sha,
                prediction_sha256=hashes, predictions=raw, controls=dict(zip(names, result)),
                method_sha256=method_hashes(), versions={"scipy": scipy.__version__, "numpy": np.__version__})
    output = args.folder / f"{args.role}_masks.json"
    print(f"Frozen {len(names)} prediction-only rows: {output} SHA256 {write_once(output, data)}", flush=True)


def score_rows(masks, labels):
    role = masks["role"]
    raw = masks["predictions"]
    results = {}
    for fold, seed, part in cases(role):
        name = key(fold, seed, part)
        label = labels[name]
        check_fields(label, fold, seed, part, role)
        if label["predictions_sha256"] != masks["prediction_sha256"][name]:
            raise ValueError(f"Labels were made with different predictions for {name}")
        trio = inputs_for(raw, fold, seed, part)
        anchor = dict(label, predictions=trio[0])
        frozen = masks["controls"][name]
        row = score108.score_anchor(anchor, trio, frozen, quotas=frozen["quota_masks"])
        if role == "target" and round(row["full"]["window"], 2) != label["refs"]["P0"]:
            raise ValueError(f"Native P0 does not reproduce the published §95 score for {name}")
        results[name] = row
    summaries = {}
    for fold in SOURCE:
        head = {s: results[key(fold, s, "head")] for s in range(9)}
        if role == "source":
            tail = {s: results[key(fold, s, "tail")] for s in range(9)}
            summaries[str(fold)] = {"head": score108.source_gate(head), "tail": score108.source_gate(tail)}
        else:
            summaries[str(fold)] = score108.target_summary(head)
    return results, summaries


def score(args):
    masks, masks_hash = read(args.folder / f"{args.role}_masks.json")
    if (masks_hash != args.masks_sha or masks["role"] != args.role or masks["prereg"] != args.prereg
            or masks["source_sha256"] != args.source_sha):
        raise ValueError("Mask fingerprint, stage or preregistration mismatch")
    labels, hashes = {}, {}
    for fold, seed, part in cases(args.role):
        name = key(fold, seed, part)
        labels[name], hashes[name] = read(args.folder / args.role / "labels" / f"{name}.json")
    results, summaries = score_rows(masks, labels)
    report = dict(role=args.role, prereg=args.prereg, masks_sha256=masks_hash, label_sha256=hashes,
                  source_sha256=args.source_sha, folds=summaries, cases=results,
                  references={k: v["refs"] for k, v in labels.items()},
                  group_references={k: v.get("group_refs", {}) for k, v in labels.items()},
                  scoring_method_sha256=method_hashes())
    if args.role == "source":
        report["advance"] = any(fold["head"]["passes"] for fold in summaries.values())
    output = args.folder / f"{args.role}_results.json"
    checksum = write_once(output, report)
    replay = dict(masks=masks, masks_sha256=masks_hash, labels=labels, label_sha256=hashes,
                  case_sha256={k: sha(serialized(v)) for k, v in results.items()},
                  folds=summaries, result_sha256=checksum)
    replay_sha = write_once(args.folder / f"{args.role}_replay.json.gz", replay)
    print(f"{output} SHA256 {checksum}; replay SHA256 {replay_sha}", flush=True)
    print(json.dumps(summaries, separators=(",", ":")), flush=True)


def replay(path):
    evidence, _ = read(path)
    masks = evidence["masks"]
    if sha(serialized(masks)) != evidence["masks_sha256"]:
        raise ValueError("Embedded mask manifest differs from its frozen fingerprint")
    for fold, seed, part in cases(masks["role"]):
        name = key(fold, seed, part)
        trio = inputs_for(masks["predictions"], fold, seed, part)
        generated = controls((*trio, masks["predictions"][name]["nframes"]))
        if generated != masks["controls"][name]:
            raise ValueError(f"Replayed geometric support, cyclic controls or quota masks differ in {name}")
    results, summaries = score_rows(masks, evidence["labels"])
    actual = {k: sha(serialized(v)) for k, v in results.items()}
    if actual != evidence["case_sha256"] or summaries != evidence["folds"]:
        raise ValueError("Replayed per-case results or fold decision differ")
    print(json.dumps(summaries, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    for op in ("masks", "score"):
        stage = subs.add_parser(op)
        stage.add_argument("role", choices=("source", "target"))
        stage.add_argument("folder", type=Path)
        stage.add_argument("--prereg", required=True)
        stage.add_argument("--source-sha", help="required for target, from both source head decisions")
        if op == "score":
            stage.add_argument("--masks-sha", required=True)
        else:
            stage.add_argument("--workers", type=int, default=1)
    rep = subs.add_parser("replay", help="verify every geometric mask, random quota, score and gate")
    rep.add_argument("package", type=Path)
    args = parser.parse_args()
    if args.stage == "replay":
        return replay(args.package)
    if re.fullmatch("[0-9a-f]{40}", args.prereg) is None:
        parser.error("--prereg must name a 40-hex commit")
    if args.role == "target":
        previous, checksum = read(args.folder / "source_results.json")
        if (not args.source_sha or args.source_sha != checksum or previous["prereg"] != args.prereg
                or previous["role"] != "source" or not previous["advance"]):
            raise ValueError("Held-out scoring requires the verified positive source decision")
    elif args.source_sha is not None:
        parser.error("--source-sha applies only to target work")
    if args.stage == "masks":
        if args.workers < 1:
            parser.error("--workers must be positive")
        freeze_masks(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
