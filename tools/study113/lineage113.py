"""Check native source event provenance in §113 against the public §108 archive."""
import gzip
import hashlib
import json
from pathlib import Path
import re


SOURCE_MASKS_SHA = "36fe1cc5470b9155a61bb8bed2f0f06242cccbbbda1b90842607976ce08e831e"


def compare108(archive_113_cases, source108_path):
    """Verify the masks, all 18 original prediction files, and shared native descriptors.

    Keys in archive_113_cases are 'fold_seed' and values are original §113 records.
    The optional §113 landmarks are irrelevant to native event identity. The
    §108 head files attest the MOT and zone digests; they identify the detector
    tag but contain no fingerprint of its weights.
    """
    public = json.loads(gzip.decompress(Path(source108_path).read_bytes()))
    masks = public["masks"]
    raw = (json.dumps(masks, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    masks_sha = hashlib.sha256(raw).hexdigest()
    if not (masks["role"] == "source" and masks_sha == public["masks_sha256"] == SOURCE_MASKS_SHA):
        raise ValueError("The §108 public source mask fingerprint differs")
    cases = [(fold, seed) for fold in (33, 35) for seed in range(9)]
    if set(archive_113_cases) != {f"{fold}_{seed}" for fold, seed in cases}:
        raise ValueError("The §113 source needs all nine seeds in both folds")

    verified = {}
    for fold, seed in cases:
        key = f"{fold}_{seed}"
        head_key = key + "_head"
        case, old = archive_113_cases[key], masks["predictions"][head_key]
        hive = 68 - fold
        clip, tag = f"BEE24-{hive}-head", f"loho_no{fold}_s{seed}"
        mot_path = f"trackers/dev/q_lk_{tag}/data/{clip}.txt"
        base = dict(fold=fold, seed=seed, clip=clip, split="dev", tag=tag, nframes=3950)
        if any(old[k] != v for k, v in base.items()):
            raise ValueError(f"§108 registered case differs at {head_key}")
        src = old["sources"]
        if set(src) != {"beetrack/zones.json", mot_path} or old["mot_sha256"] != src[mot_path]:
            raise ValueError(f"§108 native MOT path or digest differs at {head_key}")
        # §108 inputs108.save wrote insertion-ordered compact JSON without a final
        # newline. The published archive later sorted these same keys.
        table = dict(base, predictions=old["predictions"],
                     sources={"beetrack/zones.json": src["beetrack/zones.json"], mot_path: src[mot_path]},
                     mot_sha256=src[mot_path])
        raw = json.dumps(table, separators=(",", ":"), allow_nan=False).encode()
        table_sha = hashlib.sha256(raw).hexdigest()
        expected = dict(study=113, role="source", fold=fold, seed=seed, hive=hive, clip=clip, tag=tag,
                        nframes=3950, mot_path=f"runs/{mot_path}",
                        weights_path=f"runs/detect/{tag}/weights/best.pt")
        if any(case[k] != v for k, v in expected.items()):
            raise ValueError(f"§113 source metadata or relative path differs at {key}")
        hashes = case["hashes"]
        if (table_sha != masks["prediction_sha256"][head_key] or hashes["predictions108"] != table_sha
                or hashes["masks108"] != masks_sha or hashes["mot"] != src[mot_path]
                or hashes["zones"] != src["beetrack/zones.json"]
                or re.fullmatch(r"[0-9a-f]{64}", hashes["weights"]) is None):
            raise ValueError(f"§113 native provenance differs at {key}")
        descriptors = []
        for event in case["measurement"]["events"]:
            frame, direction, box = event["f_plus"], event["direction"], event["post_xyxy"]
            if (type(event["id"]) is not int or type(frame) is not int or not 1 <= frame <= 3950
                    or direction not in ("in", "out") or len(box) != 4):
                raise ValueError(f"Invalid native event descriptor at {key}")
            signed = frame if direction == "in" else -frame
            descriptors.append([event["id"], signed, *(round(v, 2) for v in box)])
        if sorted(descriptors) != old["predictions"]:
            raise ValueError(f"Native prediction ID, direction, frame or rounded box differs at {key}")
        verified[key] = table_sha
    return {"masks_sha256": masks_sha, "prediction_sha256": verified}
