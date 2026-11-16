"""Pack and replay the §116 paired crossing visibility experiment on CPU."""
import argparse
import base64
import gzip
import hashlib
import json
import re
import struct
from pathlib import Path

from frame116 import digest, pack_frame, replay_arm, require, roster_frames, wire
from summary116 import ARMS, MODES, SEEDS, summarize


PREREG = "c8fabf6c55c9737086b02050fabb3360bb5f1db2"
BOARD_ROSTER = "3f144a920a99ef438bc2552284f77bc8cab579c891e3db956280402ded62566e"
SPACES = {"original": ("native", "predictions"), "wrapped": ("wrapped", "wrapped_predictions")}


def checked(path, checksum):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == checksum, f"Unexpected SHA-256 at {path.name}")
    return json.loads(data)


def scalar_codec(x, inverse=False):
    if isinstance(x, list):
        return [scalar_codec(v, inverse) for v in x]
    if isinstance(x, dict):
        return {k: scalar_codec(v, inverse) for k, v in x.items()}
    if not inverse:
        if type(x) is float:
            return "~f" + base64.b64encode(struct.pack("<d", x)).decode("ascii")
        if isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x):
            return "~h" + base64.b64encode(bytes.fromhex(x)).decode("ascii")
        require(not isinstance(x, str) or not x.startswith(("~f", "~h")), "Unexpected scalar marker")
    elif isinstance(x, str):
        if x.startswith("~f"):
            return struct.unpack("<d", base64.b64decode(x[2:], validate=True))[0]
        if x.startswith("~h"):
            raw = base64.b64decode(x[2:], validate=True)
            require(len(raw) == 32, "Invalid SHA-256 bytes")
            return raw.hex()
    return x


def pack(root, detector_sha, score_sha):
    from receipts116 import collect_receipts, replay_receipts

    done = checked(root/"detector_done.json", detector_sha)
    index = checked(root/"scores"/"index.json", score_sha)
    require(done["prereg"] == index["prereg"] == PREREG and
            index["prediction_index_sha256"] == detector_sha, "Sources must reference the global seal")
    roster = checked(root/"roster"/"index.json", index["roster_index_sha256"])
    require(roster["prereg"] == PREREG, "Unexpected sample registration")
    geometry = checked(root/"geometry.json", index["geometry_sha256"])
    expected = {f"{p}/{c['clip']}/{c['fold']}_{s}_{a}.json" for _,p in SPACES.values()
                for c in roster["cases"] for s in SEEDS for a in ARMS}
    require(set(done["files"]) == expected and len(expected) == 48, "Incomplete joint detector inventory")
    records = {(r["clip"], str(r["seed"]), r["dose"], r["space"]):r for r in index["cases"]}
    require(len(records) == len(index["cases"]) == 48, "Incomplete official score inventory")
    receipts = collect_receipts(root, done)
    dose, _ = replay_receipts(receipts)
    sources = {"detector_index":detector_sha, "score_index":score_sha, "score_summaries":index["summaries"],
               "roster_index":index["roster_index_sha256"], "geometry":index["geometry_sha256"],
               "announced_roster":BOARD_ROSTER}
    bundle = dict(format="beetrack-study116-v1", prereg=PREREG, sources=sources,
                  receipts=receipts, clips={}, cases={}, pool=[])
    pool_index = {}
    for c in roster["cases"]:
        clip, fold = c["clip"], c["fold"]
        labels = checked(root/"roster"/c["labels_file"], c["labels_sha256"])
        request = checked(root/"roster"/c["requests_file"], c["requests_sha256"])
        require(labels["clip"] == request["clip"] == clip and
                labels["requested_frames"] == request["frames"], "Source labels/request differ")
        rect = geometry["cases"][clip]["L"]["rect"]
        require(list(rect) == list(dose[clip]["main_rect"]), "Main region does not match geometry")
        info = dict(fold=fold, nframes=labels["nframes"], events=labels["events"], polygon=labels["polygon"],
                    population=labels["population"], frames=roster_frames(labels, rect),
                    label_sha256=c["labels_sha256"], sources_sha256=c["sources_sha256"],
                    requests_sha256=c["requests_sha256"])
        bundle["clips"][clip] = info
        bundle["cases"][clip] = {}
        for seed in SEEDS:
            arm_map = bundle["cases"][clip][seed] = {}
            for arm in ARMS:
                arm_map[arm] = {}
                for mode, (space, prefix) in SPACES.items():
                    rec = records[clip, seed, arm, space]
                    name = f"{prefix}/{clip}/{fold}_{seed}_{arm}.json"
                    require(rec["prediction_sha256"] == done["files"][name], "Score parent differs")
                    pred = checked(root/name, rec["prediction_sha256"])
                    score = checked(root/"scores"/rec["file"], rec["sha256"])
                    require(pred["status"] == score["status"] == "available" and
                            pred["requests_sha256"] == c["requests_sha256"], "Incomplete observed case")
                    require(pred["predictions_sha256"] == digest(pred["predictions"]) ==
                            score["predictions_sha256"], "Score parent rows differ")
                    refs = {}
                    for f, gt in labels["gt_frames"].items():
                        frame = pack_frame(gt, pred["predictions"][f])
                        sig = digest(frame)
                        if sig not in pool_index:
                            pool_index[sig] = len(bundle["pool"])
                            bundle["pool"].append(frame)
                        refs[f] = pool_index[sig]
                    scores, _ = replay_arm(info, refs, rect, bundle["pool"], scored=score)
                    arm_map[arm][mode] = dict(frames=refs, predicted=pred["predictions_sha256"],
                                              manifest=rec["prediction_sha256"], score=rec["sha256"],
                                              observations=digest(scores))
    reports = replay(bundle, verify_result=False)
    for mode, (space, _) in SPACES.items():
        ref = index["summaries"][space]
        original = checked(root/"scores"/ref["file"], ref["sha256"])
        verify_primary(reports["visibility"], original, mode)
    bundle["result_sha256"] = digest(reports)
    return bundle


def verify_primary(summary, reference, mode):
    for case in reference["cases"]:
        result = summary["clips"][case["clip"]]["seeds"][str(case["seed"])]
        for tau, original in case["thresholds"].items():
            r = result["views"][mode][tau]["tables"]["all"]["both"]
            require(r["four_bits"] == original["four_bits"], "Official paired patterns disagree")
            require(r["h_only"]["lc"] == original["h_gained_from_u"]["lc_table"],
                    "Official high-resolution rescues disagree")
    if mode == "original":
        for clip, registered in reference["source_decisions"].items():
            result = summary["clips"][clip]
            require(result["gate"]["passes"] == registered["advance"] and
                    [result["seeds"][s]["gate"]["passes"] for s in SEEDS] == registered["seed_conditions"],
                    "Official continuation decision disagrees")


def verify_roster(bundle):
    original = {}
    for clip,c in bundle["clips"].items():
        sample = []
        for e in c["events"]:
            rank_key = f"beetrack-warp-pilot-v1|{clip}|{e['dir']}|{e['gt_tid']}|{e['arrival']}"
            require(hashlib.sha256(rank_key.encode()).hexdigest() == e["rank"], "Invalid roster rank")
            sample.append([e["rank"], e["gt_tid"], e["arrival"], e["dir"], e["before"]])
        frames = sorted({f for e in c["events"] for f in (e["arrival"], e["before"])})
        require(sorted(map(int,c["frames"])) == frames and
                digest(dict(clip=clip, nframes=c["nframes"], frames=frames)) == c["requests_sha256"],
                "Frozen sample frame set mismatch")
        original[clip] = dict(gt_sha=c["sources_sha256"]["gt_mot"], total=c["population"]["available"],
                              selected=sorted(sample), frames=frames)
    require(digest(original) == BOARD_ROSTER == bundle["sources"]["announced_roster"],
            "Reconstructed sample differs from preregistered roster seal")


def verify_indices(bundle):
    sources = bundle["sources"]
    receipts = bundle["receipts"]
    require(sources["detector_index"] == receipts["detector_sha256"] and
            sources["geometry"] == receipts["geometry_sha256"], "Official acquisition pointers disagree")
    roster_records, score_records = [], []
    clips = sorted(bundle["clips"].items(), key=lambda item: item[1]["fold"])
    for clip, info in clips:
        roster_records.append(dict(clip=clip, fold=info["fold"], frames=len(info["frames"]),
                                   selected=len(info["events"]), sources_sha256=info["sources_sha256"],
                                   labels_sha256=info["label_sha256"], requests_sha256=info["requests_sha256"],
                                   labels_file=f"{clip}/labels.private.json", requests_file=f"{clip}/requests.json"))
    require(digest(dict(format="warp-pilot-roster-index-v1", prereg=PREREG, cases=roster_records)) ==
            sources["roster_index"], "Official roster index does not reproduce")
    for mode, (space, prefix) in SPACES.items():
        for clip, info in clips:
            for seed in SEEDS:
                for arm in ARMS:
                    row = bundle["cases"][clip][seed][arm][mode]
                    suffix = f"{clip}/{info['fold']}_{seed}_{arm}.json"
                    require(row['manifest'] == receipts["detector"]["files"][f"{prefix}/{suffix}"],
                            "Score points at another prediction manifest")
                    score_records.append(dict(clip=clip, fold=info["fold"], seed=int(seed), dose=arm,
                                              space=space, file=f"{space}/{suffix}", sha256=row["score"],
                                              prediction_sha256=row["manifest"]))
    score_index = dict(format="warp-pilot-score-index-v1", prereg=PREREG,
                       prediction_index_sha256=sources["detector_index"], roster_index_sha256=sources["roster_index"],
                       geometry_sha256=sources["geometry"], cases=score_records, summaries=sources["score_summaries"])
    require(digest(score_index) == sources["score_index"], "Official score index does not reproduce")


def replay(bundle, verify_result=True):
    from receipts116 import check_manifests, check_rows, replay_receipts

    require(bundle["format"] == "beetrack-study116-v1" and bundle["prereg"] == PREREG, "Unknown study")
    verify_roster(bundle)
    verify_indices(bundle)
    dose, costs = replay_receipts(bundle["receipts"])
    observations = {}
    used = set()
    for clip, info in bundle["clips"].items():
        require(set(info["frames"]) and set(bundle["cases"][clip]) == set(SEEDS), "Incomplete source clip")
        observations[clip] = {}
        for seed in SEEDS:
            observations[clip][seed] = {}
            for arm in ARMS:
                observed = {}
                both = {}
                target = bundle["cases"][clip][seed][arm]
                require(set(target) == set(MODES), "Missing NMS view")
                for mode, e in target.items():
                    require(all(type(i) is int and 0 <= i < len(bundle["pool"]) for i in e["frames"].values()),
                            "Invalid record reference")
                    used.update(e["frames"].values())
                    observed[mode], both[mode] = replay_arm(info, e["frames"], dose[clip]["main_rect"], bundle["pool"])
                    require(digest(both[mode]) == e["predicted"] and digest(observed[mode]) == e["observations"],
                            "Replayed rows or scores differ from the seals")
                for f in info["frames"]:
                    check_rows(bundle["receipts"], clip, int(seed), arm, int(f),
                               both["original"][f], both["wrapped"][f])
                check_manifests(bundle["receipts"], clip, int(seed), arm, both["original"], both["wrapped"])
                observations[clip][seed][arm] = observed
    require(used == set(range(len(bundle["pool"]))), "Unreferenced graph")
    result = dict(visibility=summarize(bundle["clips"], observations, dose), compute=costs)
    if verify_result:
        require(digest(result) == bundle["result_sha256"], "Reconstructed public report differs")
    return result


def print_table(report):
    print("source    seed NMS       IoU   n   U   L   C   H   gain loss H-only L/C rescued")
    for clip, c in sorted(report["visibility"]["clips"].items()):
        for seed, s in c["seeds"].items():
            for mode in MODES:
                for tau, v in s["views"][mode].items():
                    r = v["tables"]["all"]["both"]
                    k = r["successes"]
                    print(clip, seed, mode, tau, r["n"], *(k[a] for a in ARMS),
                          r["u_to_l"]["gain"], r["u_to_l"]["loss"], r["h_only"]["n"],
                          f"{r['h_only']['l']}/{r['h_only']['c']}")
        print(clip, "delta J", c["dose"]["delta_j"], "passing seeds", c["gate"]["passing_seeds"],
              "advance", c["gate"]["passes"])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    make = sub.add_parser("pack")
    make.add_argument("root", type=Path)
    make.add_argument("output", type=Path)
    make.add_argument("--detector-sha", required=True)
    make.add_argument("--score-sha", required=True)
    load = sub.add_parser("replay")
    load.add_argument("archive", type=Path)
    load.add_argument("--table", action="store_true")
    load.add_argument("--report", type=Path)
    args = cli.parse_args()
    if args.command == "pack":
        bundle = pack(args.root, args.detector_sha, args.score_sha)
        compressed = gzip.compress(wire(scalar_codec(bundle)), compresslevel=9, mtime=0)
        if args.output.exists():
            require(args.output.read_bytes() == compressed, "Existing archive differs")
        else:
            args.output.write_bytes(compressed)
        print(hashlib.sha256(compressed).hexdigest(), args.output)
    else:
        with gzip.open(args.archive, "rt", encoding="utf-8") as stream:
            report = replay(scalar_codec(json.load(stream), inverse=True))
        if args.report:
            args.report.write_bytes(wire(report))
        if args.table:
            print_table(report)
        print("Verified all source rosters, forward receipts, NMS streams, frame assignments and gates.")


if __name__ == "__main__":
    main()
