"""Fit both source-only weights, then evaluate the two held-out detector folds.

Commands use the events exported by inputs.py and the clip archives from flux.py.
The independent replay interface in metrics.py needs only the resulting JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

import metrics


SOURCE_HIVE = {33: 35, 35: 33}
HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_object(path):
    data = path.read_bytes()
    return json.loads(data), digest(data)


def write_once(path, record):
    """Atomically publish a complete JSON. A retry can reuse identical contents."""
    payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(payload)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"A different artifact already exists at {path}") from None
    finally:
        tmp_path.unlink(missing_ok=True)
    return digest(payload)


def method_digests():
    # Normalize checkout line endings so public checkouts verify on Windows and Unix.
    names = ("flux.py", "inputs.py", "metrics.py", "run.py")
    return {name: digest((HERE / name).read_text(encoding="utf-8").encode()) for name in names}


def expected(role):
    if role == "source":
        return {(fold, f"BEE24-{source}-{piece}", seed)
                for fold, source in SOURCE_HIVE.items() for piece in ("head", "tail") for seed in range(3)}
    return {(fold, f"BEE24-{fold}-head", seed) for fold in SOURCE_HIVE for seed in range(9)}


def collect(inputs, role, flux_dir):
    """Join each allowed detector/seed case to an image-only, seed-independent clip."""
    if inputs["role"] != role:
        raise ValueError(f"Expected {role} events, got {inputs['role']}")
    rows = inputs["cases"]
    keys = [(row["fold"], row["clip"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected(role):
        raise ValueError(f"Incomplete, duplicate, or cross-fold {role} inputs")
    grouped = {str(fold): {} for fold in SOURCE_HIVE}
    manifests = {}
    references = {str(fold): {} for fold in SOURCE_HIVE}
    for row in sorted(rows, key=lambda r: (r["fold"], r["clip"], r["seed"])):
        fold, name, seed = str(row["fold"]), row["clip"], str(row["seed"])
        if name not in grouped[fold]:
            archive = flux_dir / f"{name}.npz"
            with np.load(archive, allow_pickle=False) as bundle:
                flux = bundle["flux"]
                meta = json.loads(str(bundle["metadata"].item()))
            nframes = 3950 if name.endswith("head") else 1000
            if flux.shape != (nframes, 2) or not np.isfinite(flux).all() or np.any(flux < 0):
                raise ValueError(f"Invalid directional flux for {name}")
            if (meta["clip"] != name or meta["nframes"] != nframes
                    or meta["fps"] != 25 or meta["columns"] != ["in", "out"]
                    or meta["source_last"] - meta["source_first"] + 1 != nframes
                    or meta["flux_f64le_sha256"] != digest(flux.astype("<f8").tobytes())):
                raise ValueError(f"Flux metadata or data hash disagrees for {name}")
            if flux[0].any():
                raise ValueError(f"First frame has a nonzero two-frame flow in {name}")
            manifests[name] = {"sha256": digest(archive.read_bytes()), "metadata": meta}
            grouped[fold][name] = {"flux": flux.tolist(), "gt": row["gt"], "p0": {}}
        clip = grouped[fold][name]
        if row["nframes"] != len(clip["flux"]) or row["gt"] != clip["gt"]:
            raise ValueError(f"Mixed lengths or GT across seeds of {name}")
        clip["p0"][seed] = row["p0"]
        if role == "target":
            references[fold][seed] = row["references"]
    return grouped, manifests, references


def freeze(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.prereg):
        raise ValueError("--prereg must be the full preregistration commit")
    raw, inputs_hash = read_object(args.source)
    sources, flux_hashes, _ = collect(raw, "source", args.flux_dir)
    fits = {fold: metrics.fit_source(clips) for fold, clips in sources.items()}
    record = {
        "role": "source_freeze", "prereg": args.prereg, "source_sha256": inputs_hash,
        "source_inputs": [{k: row[k] for k in ("fold", "clip", "seed", "tag", "p0_origin", "inputs")}
                          for row in raw["cases"]],
        "method_sha256": method_digests(), "flux_archives": flux_hashes,
        "sources": sources, "source_fit": fits,
        "frozen": {fold: fit["alpha"] for fold, fit in fits.items()},
    }
    frozen_hash = write_once(args.output, record)
    print(f"Source-only freeze {args.output}\nSHA256 {frozen_hash}\nWeights {record['frozen']}", flush=True)


def evaluate(args):
    record, frozen_hash = read_object(args.frozen)
    if frozen_hash != args.freeze_sha or record["role"] != "source_freeze":
        raise ValueError("Frozen source manifest differs from its recorded SHA-256")
    if record["method_sha256"] != method_digests():
        raise ValueError("Method source differs from the frozen implementation")
    raw, input_hash = read_object(args.target)
    grouped, archives, refs = collect(raw, "target", args.flux_dir)
    targets, results = {}, {}
    for number in SOURCE_HIVE:
        fold, name = str(number), f"BEE24-{number}-head"
        clip = grouped[fold][name]
        if archives[name] != record["flux_archives"][name]:
            raise ValueError(f"Target and opposite source reference different pixel evidence for {name}")
        if clip["gt"] != record["sources"][str(SOURCE_HIVE[number])][name]["gt"]:
            raise ValueError(f"Different source/target annotation references for the same clip {name}")
        alpha = record["frozen"][fold]
        cases = {s: metrics.score_target(clip["flux"], clip["gt"], pr, alpha)
                 for s, pr in clip["p0"].items()}
        for seed, scored in cases.items():
            if round(scored["p0"], 2) != refs[fold][seed]["P0"]:
                raise ValueError(f"Target baseline does not reproduce the published reference: {fold}/{seed}")
        targets[fold] = clip
        results[fold] = {"cases": cases, "summary": metrics.fold_summary(cases, alpha),
                         "references": refs[fold]}
    bundle = {"sources": record["sources"], "targets": targets, "frozen": record["frozen"],
              "source_freeze_sha256": frozen_hash, "target_inputs_sha256": input_hash,
              "prereg": record["prereg"], "method_sha256": record["method_sha256"],
              "flux_archives": record["flux_archives"], "source_fit": record["source_fit"],
              "source_inputs_sha256": record["source_sha256"], "source_inputs": record["source_inputs"],
              "target_inputs": [{k: row[k] for k in ("fold", "clip", "seed", "tag", "p0_origin", "inputs")}
                                for row in raw["cases"]],
              "references": refs}
    bundle_hash = write_once(args.outdir / "flow_replay.json", bundle)
    report = {"prereg": record["prereg"], "source_freeze_sha256": frozen_hash,
              "replay_sha256": bundle_hash, "source_fit": record["source_fit"],
              "frozen": record["frozen"], "folds": results}
    result_hash = write_once(args.outdir / "flow_results.json", report)
    print(f"Replay SHA256 {bundle_hash}\nResults SHA256 {result_hash}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("freeze", help="freeze both weights on their own source hive")
    fit.add_argument("source", type=Path)
    fit.add_argument("flux_dir", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--prereg", required=True)
    trial = commands.add_parser("evaluate", help="score target rows using the fingerprinted freeze")
    trial.add_argument("frozen", type=Path)
    trial.add_argument("target", type=Path)
    trial.add_argument("flux_dir", type=Path)
    trial.add_argument("outdir", type=Path)
    trial.add_argument("--freeze-sha", required=True)
    args = ap.parse_args()
    (freeze if args.command == "freeze" else evaluate)(args)


if __name__ == "__main__":
    main()
