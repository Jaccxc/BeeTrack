"""Prepare immutable image caches, freeze both source folds, then score targets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import torch

import strip_data
import strip_model
import strip_score


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for name in
            ("run_boundary.py", "strip_data.py", "strip_model.py", "strip_score.py")}


def cache_folder(args, clip, part):
    return args.out / "cache" / f"BEE24-{clip}-{part}"


def load_cache(args, clip, part):
    folder = cache_folder(args, clip, part)
    receipt = read_json(folder / "receipt.json")
    if receipt["clip"] != clip or receipt["part"] != part or receipt["prereg"] != args.prereg:
        raise ValueError("Cache receipt identity mismatch")
    if digest(folder / "images.npy") != receipt["cache_sha256"]:
        raise ValueError("Cache hash mismatch")
    if digest(args.repo / "beetrack/zones.json") != receipt["zones_sha256"]:
        raise ValueError("Polygon hash mismatch")
    if receipt["strip_data_sha256"] != code_hashes()["strip_data.py"]:
        raise ValueError("Cache extraction implementation changed")
    return np.load(folder / "images.npy", mmap_mode="r", allow_pickle=False), receipt


def prepare(args):
    destination = cache_folder(args, args.clip, args.part)
    if destination.exists():
        _, receipt = load_cache(args, args.clip, args.part)
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "cache"
        stage.mkdir()
        bank = strip_data.cache_images(args.repo, f"BEE24-{args.clip}", args.part, stage / "images.npy")
        shape = list(bank.shape)
        del bank
        receipt = dict(clip=args.clip, part=args.part, prereg=args.prereg, shape=shape,
                       cache_sha256=digest(stage / "images.npy"),
                       zones_sha256=digest(args.repo / "beetrack/zones.json"),
                       strip_data_sha256=code_hashes()["strip_data.py"])
        write_json(stage / "receipt.json", receipt)
        stage.rename(destination)
    return receipt


def truth(args, clip, part):
    split = "dev" if part == "head" else "val"
    path = args.repo / "runs/gt" / split / f"BEE24-{clip}-{part}" / "gt/gt.txt"
    return strip_data.event_cells(args.repo, f"BEE24-{clip}", part), digest(path)


def freeze(args):
    source, target = args.source, 68 - args.source
    destination = args.out / f"fold_source{source}"
    if destination.exists():
        raise FileExistsError(f"Refusing retraining or overwrite: {destination}")
    head, head_receipt = load_cache(args, source, "head")
    tail, tail_receipt = load_cache(args, source, "tail")
    target_bank, target_receipt = load_cache(args, target, "head")
    head_events, head_gt_sha = truth(args, source, "head")
    tail_events, tail_gt_sha = truth(args, source, "tail")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"freeze{source}-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "fold"
        stage.mkdir()
        arms = {}
        for arm in ("T", "S"):
            model = strip_model.train_model(head, head_events, 1, len(head), arm)
            calibration = strip_score.calibrate(
                strip_model.predict_model(model, tail, 1, len(tail), arm), tail_events)
            threshold = calibration["threshold"]
            scores = strip_model.predict_model(model, target_bank, 1, len(target_bank), arm)
            torch.save(model.state_dict(), stage / f"{arm}_state.pt")
            np.save(stage / f"{arm}_scores.npy", scores, allow_pickle=False)
            write_json(stage / f"{arm}_events.json", strip_score.decode(scores, threshold))
            write_json(stage / f"{arm}_calibration.json", calibration)
            arms[arm] = dict(threshold=threshold)
        files = {p.name: digest(p) for p in stage.iterdir()}
        receipt = dict(prereg=args.prereg, source=source, target=target,
                       source_head=head_receipt, source_tail=tail_receipt, target_head=target_receipt,
                       source_head_gt_sha256=head_gt_sha, source_tail_gt_sha256=tail_gt_sha,
                       target_nframes=len(target_bank), code=code_hashes(), arms=arms, files=files)
        write_json(stage / "receipt.json", receipt)
        stage.rename(destination)
    return receipt


def frozen(args, source):
    folder = args.out / f"fold_source{source}"
    receipt = read_json(folder / "receipt.json")
    if (receipt["source"] != source or receipt["target"] != 68-source
            or receipt["prereg"] != args.prereg or receipt["code"] != code_hashes()):
        raise ValueError("Freeze identity or implementation mismatch")
    for name, sha in receipt["files"].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError("Frozen artifact hash mismatch")
    return folder, receipt


def score(args):
    destination = args.out / "summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    # Verify both receipts and every frozen artifact before opening either target GT.
    folds = [frozen(args, source) for source in (33, 35)]
    summary = dict(prereg=args.prereg, metric="time-cell F1, tolerance 5 frames / 1 circular cell; not official box F1", folds=[])
    for folder, receipt in folds:
        gt, gt_sha = truth(args, receipt["target"], "head")
        row = dict(source=receipt["source"], target=receipt["target"],
                   freeze_receipt_sha256=digest(folder / "receipt.json"),
                   target_gt_sha256=gt_sha, gt_events=gt, arms={})
        for arm in ("T", "S"):
            events = read_json(folder / f"{arm}_events.json")
            scores = np.load(folder / f"{arm}_scores.npy", allow_pickle=False)
            if scores.shape != (receipt["target_nframes"], 32, 2):
                raise ValueError("Frozen score shape mismatch")
            decoded = strip_score.decode(scores, receipt["arms"][arm]["threshold"])
            if [list(e) for e in decoded] != events:
                raise ValueError("Frozen decode mismatch")
            result = strip_score.counts(events, gt)
            for name in ("in", "out"):
                result[name].update(fp=result[name]["pred"]-result[name]["tp"],
                                    fn=result[name]["gt"]-result[name]["tp"])
            result.update(fp=result["pred"]-result["tp"], fn=result["gt"]-result["tp"],
                          events=events, threshold=receipt["arms"][arm]["threshold"],
                          err=strip_score.window_err(events, gt, len(scores)))
            result["W"] = 100*result["err"]/len(gt) if gt else None
            row["arms"][arm] = result
        summary["folds"].append(row)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.out, prefix="summary-", suffix=".json", delete=False) as temp:
        json.dump(summary, temp, indent=2, allow_nan=False)
        temporary = Path(temp.name)
    temporary.rename(destination)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "score"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--clip", type=int, choices=(33, 35))
    parser.add_argument("--part", choices=("head", "tail"))
    parser.add_argument("--source", type=int, choices=(33, 35))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.prereg):
        parser.error("--prereg must be the full lowercase 40-character SHA")
    args.repo, args.out = args.repo.resolve(), args.out.resolve()
    if args.command == "prepare" and (args.clip is None or args.part is None):
        parser.error("prepare requires --clip and --part")
    if args.command == "freeze" and args.source is None:
        parser.error("freeze requires --source")
    sys.path.insert(0, str(args.repo))
    result = {"prepare": prepare, "freeze": freeze, "score": score}[args.command](args)
    if args.command == "score":
        for fold in result["folds"]:
            for arm, metrics in fold["arms"].items():
                print(fold["target"], arm, {k: metrics[k] for k in ("tp", "fp", "fn", "f1", "W", "threshold")})
    else:
        print(args.command, args.clip or args.source, args.part or "both arms", "sealed", result["prereg"])


if __name__ == "__main__":
    main()
