"""Pack and replay the source-frozen §107 timing study without any original video."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

import metrics


OTHER = {"33": "35", "35": "33"}
SCHEMA = "BeeTrack study 107 / four clips / 1"


def encoded(obj):
    """Byte format used by the freeze and evaluation driver."""
    return (json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def split_clips(bundle):
    # Keep source and target P0 predictions separate even for the same physical head.
    shared = {}

    def save(name, record):
        flux = np.asarray(record["flux"], dtype="<f8")
        signal = {"flux": {"dtype": "<f8", "shape": list(flux.shape),
                           "base64": base64.b64encode(flux.tobytes()).decode("ascii")}, "gt": record["gt"]}
        if name in shared and shared[name] != signal:
            raise ValueError(f"GT or flux differs between roles for {name}")
        shared[name] = signal
        return {"clip": name, "p0": record["p0"]}

    study = bundle.copy()
    study["sources"] = {h: {k: save(k, v) for k, v in clips.items()} for h, clips in bundle["sources"].items()}
    study["targets"] = {h: save(f"BEE24-{h}-head", row) for h, row in bundle["targets"].items()}
    return dict(schema=SCHEMA, replay_sha256=sha(encoded(bundle)), clips=shared, study=study)


def unpack(record):
    if record["schema"] != SCHEMA:
        raise ValueError("Unsupported study data format")
    clips, study = record["clips"], record["study"].copy()
    expected = {f"BEE24-{h}-{suffix}" for h in OTHER for suffix in ("head", "tail")}
    if set(clips) != expected or set(study["sources"]) != set(OTHER) or set(study["targets"]) != set(OTHER):
        raise ValueError("Incomplete or unexpected study inputs")
    streams = {}
    for clip, stream in clips.items():
        nframes = 3950 if clip.endswith("-head") else 1000
        matrix = stream["flux"]
        if matrix["dtype"] != "<f8" or matrix["shape"] != [nframes, 2]:
            raise ValueError(f"Wrong array dtype/shape for {clip}")
        raw = base64.b64decode(matrix["base64"], validate=True)
        array = np.frombuffer(raw, dtype="<f8").reshape(nframes, 2)
        meta = study["flux_archives"][clip]["metadata"]
        if (array.shape != (nframes, 2) or meta["nframes"] != nframes or meta["columns"] != ["in", "out"]
                or not np.isfinite(array).all() or (array < 0).any() or array[0].any()
                or sha(array.tobytes()) != meta["flux_f64le_sha256"]):
            raise ValueError(f"Flux provenance mismatch for {clip}")
        streams[clip] = {"flux": array.tolist(), "gt": stream["gt"]}

    def expand(name, row):
        if row["clip"] != name:
            raise ValueError("A fold referenced an unexpected clip")
        return {**streams[name], "p0": row["p0"]}

    for h, other in OTHER.items():
        if set(study["sources"][h]) != {f"BEE24-{other}-head", f"BEE24-{other}-tail"}:
            raise ValueError(f"Unexpected source for held-out hive {h}")
    study["sources"] = {h: {k: expand(k, v) for k, v in rows.items()} for h, rows in study["sources"].items()}
    study["targets"] = {h: expand(f"BEE24-{h}-head", row) for h, row in study["targets"].items()}
    if sha(encoded(study)) != record["replay_sha256"]:
        raise ValueError("Reconstructed full archive differs from the experiment driver's bundle")
    frozen = dict(role="source_freeze", prereg=study["prereg"], source_sha256=study["source_inputs_sha256"],
                  source_inputs=study["source_inputs"], method_sha256=study["method_sha256"],
                  flux_archives=study["flux_archives"], sources=study["sources"],
                  source_fit=study["source_fit"], frozen=study["frozen"])
    if sha(encoded(frozen)) != study["source_freeze_sha256"]:
        raise ValueError("Source-only freeze differs from its preregistered-run fingerprint")
    return study


def compute(study, replay_sha):
    output = metrics.replay(study)
    folds = {}
    for hive, target in output["targets"].items():
        cases = {str(seed): result for seed, result in target["seeds"].items()}
        refs = study["references"][hive]
        for seed, row in cases.items():
            if round(row["p0"], 2) != refs[seed]["P0"]:
                raise ValueError(f"Original P0 baseline disagrees for {hive}/{seed}")
        for key in ("candidate_scores", "candidates"):
            np.testing.assert_allclose(output["sources"][hive][key], study["source_fit"][hive][key],
                                       atol=1e-9, rtol=0)
        for clip, scores in output["sources"][hive]["scores_by_clip_seed_candidate"].items():
            np.testing.assert_allclose(scores, study["source_fit"][hive]["scores_by_clip_seed_candidate"][clip],
                                       atol=1e-9, rtol=0)
        folds[hive] = dict(cases=cases, summary=target["summary"], references=refs)
    return dict(prereg=study["prereg"], source_freeze_sha256=study["source_freeze_sha256"],
                replay_sha256=replay_sha, source_fit=output["sources"], frozen=study["frozen"], folds=folds)


def print_tables(study, report):
    for hive, result in report["folds"].items():
        fit = report["source_fit"][hive]
        print(f"\nHeld out {hive}; source alpha {fit['alpha']}; candidate errors for {fit['candidates']}")
        print("source_clip seed   alpha=0  alpha=.25  alpha=.5")
        for clip, scores in fit["scores_by_clip_seed_candidate"].items():
            for s, row in enumerate(scores):
                print(clip, s, *(f"{v:.2f}" for v in row))
        print("equal-clip mean", *(f"{v:.2f}" for v in fit["candidate_scores"]))
        cases = result["cases"]
        for name in ("primary", "fixed"):
            print(f"\n{name}    seed  P0  aligned  uniform  swapped  phase_median  shifts_beaten/157  joint  floor")
            for s, row in cases.items():
                arm = row[name]
                print(s, f"{row['p0']:.2f}", *(f"{arm[k]:.2f}" for k in ("aligned", "uniform", "swapped", "phase_median")),
                      arm["phase_strictly_beaten"], int(arm["beats_uniform_and_phase"]), f"{row['floor']:.2f}")
            arm = result["summary"][name]
            print(f"mean aligned/uniform/swapped: {arm['aligned']:.2f}/{arm['uniform']:.2f}/{arm['swapped']:.2f}; "
                  f"gain vs P0 {arm['gain_vs_p0']:.2f} pp; joint wins {arm['joint_wins']}/9; "
                  f"gain vs swapped {arm['gain_vs_swapped']:.2f} pp in {arm['directional_wins']}/9")
        primary = result["summary"]["primary"]
        print(f"Registered main gate: {primary['passes']}; directional gate: {primary['directional_passes']}")
        print("\nseed   P0   primary   ByteTrack   calibrated_ByteTrack   source_selected_TOPICTrack")
        for s, row in cases.items():
            ref = result["references"][s]
            top = ref.get("TOPICTrack selected")
            print(s, f"{row['p0']:.2f}", f"{row['primary']['aligned']:.2f}", f"{ref['ByteTrack']:.2f}",
                  f"{ref['calibrated ByteTrack']:.2f}", "—" if top is None else f"{top:.2f}")
        target = study["targets"][hive]
        n = len(target["flux"])
        gt = metrics.event_mass(target["gt"], n)
        g = metrics.window_counts(gt)
        flat = metrics.window_counts(np.repeat((gt.sum(axis=0) / n)[None, :], n, axis=0))
        flat_error = 100.0 * np.abs(flat - g).sum() / g.sum()
        q = np.asarray(target["flux"])
        total = q.sum(axis=0)
        variation = (0.5 * np.abs(np.diff(q / total, axis=1)).sum()) if np.min(total) > 1e-12 else None
        print(f"Mean P0 {result['summary']['p0_mean']:.2f}; mean floor {result['summary']['floor_mean']:.2f}; "
              f"flat, true direction totals {flat_error:.2f} (published §94: {22.04 if hive == '33' else 23.29:.2f}); "
              f"total variation of inward/outward timing {variation if variation is not None else 'undefined'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="cmd", required=True)
    pack = subs.add_parser("pack", help="deduplicate the canonical evaluated bundle and validate the driver result")
    pack.add_argument("driver_replay", type=Path)
    pack.add_argument("driver_results", type=Path)
    pack.add_argument("output", type=Path)
    replay = subs.add_parser("replay", help="verify fingerprints and reproduce both folds")
    replay.add_argument("archive", type=Path)
    replay.add_argument("--json", type=Path, help="save every phase, control, source fit, and baseline reference")
    args = ap.parse_args()
    if args.cmd == "pack":
        payload = args.driver_replay.read_bytes()
        packed = split_clips(json.loads(payload))
        if packed["replay_sha256"] != sha(payload):
            raise ValueError("Expected the original canonical driver output")
        study = unpack(packed)
        result = compute(study, packed["replay_sha256"])
        if result != json.loads(args.driver_results.read_bytes()):
            raise ValueError("Recomputed fits, 18 targets, baseline references or phase controls disagree")
        data = json.dumps(packed, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
        args.output.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
        print(f"Verified both folds. Archive SHA-256: {sha(args.output.read_bytes())}")
    else:
        packed = json.loads(gzip.decompress(args.archive.read_bytes()))
        study = unpack(packed)
        result = compute(study, packed["replay_sha256"])
        print_tables(study, result)
        if args.json is not None:
            args.json.write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False),
                                 encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
