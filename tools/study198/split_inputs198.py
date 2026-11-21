"""One-time data organization of checked phase1 graphs into four hive shards.

No model fitting, marginal labels, target scores, pixels, GT loaders or native
MOT reconstruction. The independent process alone opens the all-hive graph file.
Training loaders open only the explicitly allowed three shard payloads.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

DOMAINS = ("35", "33", "W", "G")
SCHEMA = "endpoint198-source-graph-shards-v1"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")


def split_checked(checked, output, *, prereg, stage1_receipt):
    if checked["checked_streams"] != 60 or checked["checked_arms"] != 420 or len(checked["checks"]) != 420:
        raise ValueError("the checked60stream/420arm graph package is required")
    grouped = {h: {} for h in DOMAINS}
    count = 0
    for unit, members in checked["models"].items():
        if len(members) != 3: raise ValueError("three members per source group required")
        domains = {model["context"]["domain"] for model in members.values()}
        if len(domains) != 1 or next(iter(domains)) not in DOMAINS: raise ValueError("mixed/unknown source hive")
        domain = next(iter(domains))
        for member, model in members.items():
            if model["context"]["unit"] != unit or str(model["context"]["member"]) != str(member):
                raise ValueError("graph context differs from its source key")
            if model["schema"] != "endpoint198-full-scene-state-graph-v1": raise ValueError("wrong checked graph model")
            count += 1
        grouped[domain][unit] = members
    if count != 60 or len(checked["models"]) != 20 or any(not units for units in grouped.values()):
        raise ValueError("fixed four-domain20group/60member inventory required")
    root = Path(output)
    if root.exists(): raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".source-shards198-", dir=root.parent) as tmp:
        work = Path(tmp)/"payload"; work.mkdir()
        shards = {}
        for hive in DOMAINS:
            name = hive+".json"
            # Only the self-contained graph models and their existing context.
            save(work/name, dict(schema=SCHEMA, prereg=prereg, hive=hive, models=grouped[hive]))
            shards[hive] = dict(file=name, sha256=sha(work/name), units=len(grouped[hive]), members=3*len(grouped[hive]))
        manifest = dict(schema=SCHEMA, prereg=prereg, stage1_receipt=stage1_receipt, shards=shards,
                        splitter_code_sha256=sha(__file__))
        save(work/"manifest.json", manifest)
        (work/"SEALED.sha256").write_text(sha(work/"manifest.json")+"\n")
        work.rename(root)
    return dict(manifest_sha256=sha(root/"manifest.json"), stage1_receipt=stage1_receipt, shards=shards)


def split_stage1(stage1, output, *, expected_manifest, expected_graph):
    stage1 = Path(stage1)
    digest = sha(stage1/"manifest.json")
    if digest != expected_manifest or (stage1/"SEALED.sha256").read_text().strip() != digest:
        raise ValueError("finalized phase1 manifest differs")
    manifest = json.loads((stage1/"manifest.json").read_text())
    for name in ("receipt.json", "capacity.json", "graph_data.json"):
        if sha(stage1/name) != manifest[name]: raise ValueError("final phase1 file changed")
    if manifest["graph_data.json"] != expected_graph: raise ValueError("checked graph digest differs")
    receipt = json.loads((stage1/"receipt.json").read_text())
    if receipt["graph_checked_arms"] != 420 or receipt["physical_verified_members"] != 60 or receipt["graph_data_sha256"] != expected_graph:
        raise ValueError("phase1 native/graph verification incomplete")
    capacity = json.loads((stage1/"capacity.json").read_text())
    if capacity.get("all60_native_graph_verified") is not True: raise ValueError("phase1 incomplete")
    checked = json.loads((stage1/"graph_data.json").read_text())
    binding = dict(manifest_sha256=digest, graph_data_sha256=expected_graph, capacity_sha256=manifest["capacity.json"],
                   receipt_sha256=manifest["receipt.json"], peer_data_sha256=capacity["input_sha256"])
    return split_checked(checked, output, prereg=capacity["prereg"], stage1_receipt=binding)


def manifest_only(root):
    root = Path(root)
    digest = sha(root/"manifest.json")
    if (root/"SEALED.sha256").read_text().strip() != digest: raise ValueError("source shard manifest differs")
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["schema"] != SCHEMA or set(manifest["shards"]) != set(DOMAINS): raise ValueError("incomplete source shard roster")
    return manifest, digest


def load_source_shards(root, heldout, *, source_domains, opener=None):
    allowed = tuple(source_domains)
    if heldout not in DOMAINS or len(allowed) != 3 or set(allowed) != set(DOMAINS)-{heldout}:
        raise ValueError("explicit exactly-other-three source whitelist required")
    root = Path(root)
    manifest, digest = manifest_only(root)  # hashes only metadata, never heldout payload
    read = opener or (lambda path: path.read_bytes())
    models, proofs = {}, {}
    for hive in allowed:
        proof = manifest["shards"][hive]
        if proof["file"] != hive+".json": raise ValueError("unsafe/mislabelled source shard name")
        raw = read(root/proof["file"])
        if hashlib.sha256(raw).hexdigest() != proof["sha256"]: raise ValueError("source shard changed")
        payload = json.loads(raw)
        if payload["hive"] != hive or payload["prereg"] != manifest["prereg"] or payload["schema"] != SCHEMA:
            raise ValueError("source shard registration/domain differs")
        for unit, members in payload["models"].items():
            if unit in models or any(m["context"]["domain"] != hive for m in members.values()): raise ValueError("cross-hive shard context")
            models[unit] = members
        proofs[hive] = proof
    return dict(models=models, source_hives=allowed, shard_receipts=proofs, manifest_sha256=digest,
                prereg=manifest["prereg"], stage1_receipt=manifest["stage1_receipt"])


def self_test():
    checked = dict(checked_streams=60, checked_arms=420, checks=[{}]*420, models={})
    for hive, groups in (("35", 4), ("33", 4), ("W", 6), ("G", 6)):
        for group in range(groups):
            unit = hive+str(group)
            checked["models"][unit] = {str(m): dict(schema="endpoint198-full-scene-state-graph-v1",
                context=dict(domain=hive, unit=unit, member=str(m)), mock_graph="no labels/fit computed") for m in range(3)}
    with TemporaryDirectory(prefix="source-shards198-") as tmp:
        root = Path(tmp)/"sources"
        split_checked(checked, root, prereg="synthetic", stage1_receipt=dict(manifest_sha256="synthetic", graph_data_sha256="synthetic"))
        for heldout in DOMAINS:
            calls = []
            def reader(path):
                calls.append(path.name)
                assert path.name != heldout+".json"
                return path.read_bytes()
            allowed = tuple(h for h in DOMAINS if h != heldout)
            loaded = load_source_shards(root, heldout, source_domains=allowed, opener=reader)
            assert set(calls) == {h+".json" for h in allowed}
            assert all(m["context"]["domain"] != heldout for members in loaded["models"].values() for m in members.values())
        # The heldout payload can be physically absent and is not opened/hash-read.
        (root/"35.json").unlink()
        load_source_shards(root, "35", source_domains=("33", "W", "G"))
        spy = []
        try: load_source_shards(root, "35", source_domains=("35", "W", "G"), opener=lambda p: spy.append(p))
        except ValueError: pass
        else: raise AssertionError("heldout shard access accepted")
        assert not spy
    return dict(data_only_split="60 graphs, no labels/models fitted", four_source_fold_spies="passed",
                heldout_file_absent="passed", illegal_whitelist="rejected before any payload read")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--stage1", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest-sha")
    parser.add_argument("--graph-sha")
    args = parser.parse_args()
    if args.self_test: print(json.dumps(self_test(), indent=2))
    elif all((args.stage1, args.out, args.manifest_sha, args.graph_sha)):
        print(json.dumps(split_stage1(args.stage1, args.out, expected_manifest=args.manifest_sha, expected_graph=args.graph_sha), indent=2))
    else: parser.error("use --self-test or explicit completed phase1/output/hash arguments")
