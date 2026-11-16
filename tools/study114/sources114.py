"""Validate source-side inputs as bytes, without loading detections, tracks or GT rows."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory


ARCHIVES = {
    "docs/study105_results.json": "6c6c455b54023b3f87cc957c205122db552133c21fb389c1d6423d345e314541",
    "docs/study108_source.json.gz": "2a7d8bf2d4b027dafa59d924e5b908ff2b88c0edbdf9fcded097c60383edd383",
    "docs/study113_replay.json.gz": "f646253f846167406f28380e7013fef733d26227f6658a714a55b36d53772e65",
}
FIELDS = ("det", "mot", "gt", "zones")
SOURCE_INPUTS_SHA = "39b2a95c1c85acebf11194967a495e536ddfb893b96f28c473d35f17a2672ecd"


def wire(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_source(repo_root, study108_root, logical_path):
    """Resolve a metadata path. The external §108 root holds its source tail MOTs."""
    p = PurePosixPath(logical_path)
    if p.is_absolute() or ".." in p.parts or "\\" in logical_path or ":" in logical_path:
        raise ValueError(f"Expected a relative logical path: {logical_path}")
    if p.parts[:3] == ("study108", "source", "tail_mot"):
        return Path(study108_root).joinpath(*p.parts[1:])
    if p.parts[:2] not in (("runs", "gt"), ("runs", "trackers"), ("runs", "dets")) and str(p) != "beetrack/zones.json":
        raise ValueError(f"Unknown source type: {logical_path}")
    return Path(repo_root).joinpath(*p.parts)


def _archives(repo_root):
    data = []
    for filename, expected in ARCHIVES.items():
        raw = (Path(repo_root) / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Public historical archive changed: {filename}")
        data.append(json.loads(gzip.decompress(raw) if filename.endswith(".gz") else raw))
    return data


def expected_metadata(repo_root):
    """Return metadata with historical pins. Six source-head detector pins are initially null."""
    s105, s108, s113 = _archives(repo_root)
    # §105 used the opposite detector fold but the identical repaired head GT.
    gt105 = {}
    for row in s105["cases"]:
        path = f"gt/dev/BEE24-{row['hive']}-head/gt/gt.txt"
        value = row["sources"][path]
        if gt105.setdefault(path, value) != value:
            raise ValueError("Conflicting §105 head GT fingerprints")
    cases = {}
    for fold in (33, 35):
        hive = 68-fold
        for seed in range(3):
            tag = f"loho_no{fold}_s{seed}"
            for part, split, nframes in (("head", "dev", 3950), ("tail", "val", 1000)):
                key, clip = f"{fold}_{seed}_{part}", f"BEE24-{hive}-{part}"
                meta = dict(fold=fold, hive=hive, seed=seed, part=part, clip=clip,
                            split=split, nframes=nframes, tag=tag)
                reference = s108["masks"]["predictions"][key]
                label = s108["labels"][key]
                for field in ("fold", "seed", "clip", "split", "nframes", "tag"):
                    if reference[field] != meta[field] or label[field] != meta[field]:
                        raise ValueError(f"§108 case identity mismatch: {key}/{field}")
                det = f"dets/{tag}/{split}/{clip}.npy"
                track = f"trackers/{split}/q_lk_{tag}/data/{clip}.txt"
                gt = f"gt/{split}/{clip}/gt/gt.txt"
                zones = "beetrack/zones.json"
                paths = dict(det="runs/"+det, mot="runs/"+track if part=="head" else
                             f"study108/source/tail_mot/{key}.txt", gt="runs/"+gt, zones=zones)
                src = reference["sources"]
                hashes = dict(det=src.get(det), mot=reference["mot_sha256"],
                              gt=label["gt_mot_sha256"], zones=src[zones])
                provenance = dict(det=["108"], mot=["108"], gt=["108"], zones=["108"])
                if part == "head":
                    r = s113["cases"][f"{fold}_{seed}"]["meta"]
                    if (any(r[k] != meta[k] for k in ("hive", "tag", "nframes", "clip", "seed", "fold"))
                            or r["role"] != "source" or r["mot_path"] != paths["mot"]
                            or r["hashes"]["mot"] != src[track] or src[track] != hashes["mot"]
                            or r["hashes"]["zones"] != hashes["zones"] or gt105[gt] != hashes["gt"]):
                        raise ValueError(f"Historical source head fingerprints disagree: {key}")
                    provenance["det"] = ["114 first local hash"]
                    provenance["mot"].append("113")
                    provenance["gt"].append("105")
                    provenance["zones"].append("113")
                elif hashes["det"] is None:
                    raise ValueError(f"§108 lacks a tail detector fingerprint: {key}")
                cases[key] = dict(meta, paths=paths, hashes=hashes, provenance=provenance)
    return dict(study=114, role="source", archives=dict(ARCHIVES), cases=cases)


def verify_metadata(data, repo_root):
    """Check portable paths, historical hashes and metadata without local raw source files."""
    expected = expected_metadata(repo_root)
    for name, row in expected["cases"].items():
        if row["hashes"]["det"] is None:
            supplied = data["cases"][name]["hashes"]["det"]
            if not isinstance(supplied, str) or not re.fullmatch("[0-9a-f]{64}", supplied):
                raise ValueError(f"Invalid locally measured detector fingerprint: {name}")
            row["hashes"]["det"] = supplied
    if data != expected:
        raise ValueError("Source metadata, historical fingerprints or registered cases differ")
    if hashlib.sha256(wire(data)).hexdigest() != SOURCE_INPUTS_SHA:
        raise ValueError("Source metadata differs from the initial §114 byte-hash preflight")


def preflight(repo_root, study108_root):
    """Stream source bytes once per path. Return only public metadata and digests."""
    data = expected_metadata(repo_root)
    measured = {}
    for name, row in data["cases"].items():
        for field in FIELDS:
            path = row["paths"][field]
            if path not in measured:
                measured[path] = digest_file(resolve_source(repo_root, study108_root, path))
            expected = row["hashes"][field]
            if expected is not None and measured[path] != expected:
                raise ValueError(f"Historical fingerprint mismatch: {name}/{field}")
            row["hashes"][field] = measured[path]
    return data


def save(output, data):
    raw = wire(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        partial = Path(tmp)/"manifest"
        partial.write_bytes(raw)
        try:
            os.link(partial, output)
        except FileExistsError:
            if output.read_bytes() != raw:
                raise ValueError("A different source manifest already exists")
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--study108-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = preflight(args.repo, args.study108_root)
    verify_metadata(manifest, args.repo)
    print(f"{save(args.output, manifest)} {args.output.name}")
